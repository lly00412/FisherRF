#
# Standalone 3DGS training with a per-Gaussian uncertainty parameter u.
#
# u is supervised by VCURF uncertainty maps, following the confidence loss of
# CoMe (Radl et al., arXiv 2603.24725, Eq 9-11):
#   gamma_i = exp(s_i),  u_i = exp(-s_i)
#   C_hat(p) = sum_i gamma_i a_i T_i / max(sum_i a_i T_i, eps)              (Eq 9)
#   L_u = mean_p [ C_hat(p) * sg(D~(p)) - beta * log C_hat(p) ]           (Eq 10, VCURF replaces L_pp)
#   D~ = beta * D / median(D),  D = d_l2 / median(d_l2) + c_l2 / median(c_l2)
# The optimum is C_hat = 1 / (D / median(D)), i.e. rendered u is proportional to VCURF.
#
# The VCURF loss only updates s. Position, opacity, scale and rotation are
# detached in the confidence pass, so geometry receives no gradient from it.
# No CUDA changes: C_hat is rendered with the existing RGBD rasterizer in a
# second pass using colors_precomp = [gamma, 1, 0] and a black background.
#
# This file does not modify any existing module.
#

import os
import sys
import math
import uuid
from random import randint
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.cameras import VirtualCam
from utils.general_utils import safe_state, build_rotation
from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import BackwardWarping, extract_scene_center_and_C2W
from utils.system_utils import mkdir_p
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization_w_depth import GaussianRasterizer as RGBDGaussianRasterizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ---------------------------------------------------------------------------
# Gaussian model with an extra per-Gaussian log-confidence s (stored as _unc)
# ---------------------------------------------------------------------------
class GaussianModelWithUnc(GaussianModel):

    def __init__(self, sh_degree: int, unc_lr: float = 1e-3, c_min: float = 1e-3):
        super().__init__(sh_degree)
        self._unc = torch.empty(0)
        # True once a Gaussian has received any gradient from the VCURF loss.
        # Gaussians never supervised (only seen by masked pixels) report the max uncertainty.
        self.unc_observed = torch.empty(0, dtype=torch.bool)
        self.unc_lr = unc_lr
        self.c_min = c_min  # C_hat lower clamp; u_max = 1 / c_min

    @property
    def u_max(self):
        return 1.0 / self.c_min

    # gamma = exp(s): confidence used for the loss (carries grad)
    @property
    def get_confidence(self):
        return torch.exp(self._unc)

    # u = exp(-s) for supervised Gaussians, u_max for never-supervised ones
    @property
    def get_uncertainty(self):
        u = torch.exp(-self._unc)
        return torch.where(self.unc_observed[:, None], u, torch.full_like(u, self.u_max))

    @torch.no_grad()
    def mark_observed(self, weight_on_supervised_pixels, min_weight):
        """Mark Gaussians whose blending weight on supervised pixels exceeds min_weight."""
        self.unc_observed |= weight_on_supervised_pixels > min_weight

    def _zeros_unc(self, n):
        return torch.zeros((n, 1), dtype=torch.float, device="cuda")

    def _false_obs(self, n):
        return torch.zeros((n,), dtype=torch.bool, device="cuda")

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        self._unc = nn.Parameter(self._zeros_unc(self._xyz.shape[0]).requires_grad_(True))
        self.unc_observed = self._false_obs(self._xyz.shape[0])

    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.optimizer.add_param_group({'params': [self._unc], 'lr': self.unc_lr, "name": "unc"})

    def capture(self):
        return super().capture() + (self._unc, self.unc_observed)

    def restore(self, model_args, training_args):
        # _unc must exist before training_setup builds the optimizer groups
        self._unc = model_args[-2]
        self.unc_observed = model_args[-1]
        super().restore(model_args[:-2], training_args)

    # ---- ply io ----
    def construct_list_of_attributes(self):
        return super().construct_list_of_attributes() + ['unc', 'unc_observed']

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        unc = self._unc.detach().cpu().numpy()
        observed = self.unc_observed.float()[:, None].cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, unc, observed), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        super().load_ply(path)
        plydata = PlyData.read(path)
        names = [p.name for p in plydata.elements[0].properties]
        if 'unc' in names:
            unc = np.asarray(plydata.elements[0]['unc'])[..., np.newaxis]
            unc = torch.tensor(unc, dtype=torch.float, device="cuda")
        else:
            unc = self._zeros_unc(self._xyz.shape[0])
        self._unc = nn.Parameter(unc.requires_grad_(True))
        if 'unc_observed' in names:
            obs = np.asarray(plydata.elements[0]['unc_observed']) > 0.5
            self.unc_observed = torch.tensor(obs, dtype=torch.bool, device="cuda")
        else:
            self.unc_observed = self._false_obs(self._xyz.shape[0])

    # ---- prune / densify: same as GaussianModel, plus _unc ----
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._unc = optimizable_tensors["unc"]
        self.unc_observed = self.unc_observed[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_scaling, new_rotation, new_unc=None, new_observed=None):
        if new_unc is None:  # e.g. add_random_gaussians
            new_unc = self._zeros_unc(new_xyz.shape[0])
        if new_observed is None:
            new_observed = self._false_obs(new_xyz.shape[0])
        self.unc_observed = torch.cat((self.unc_observed, new_observed), dim=0)
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "unc": new_unc}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._unc = optimizable_tensors["unc"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_unc = self._unc[selected_pts_mask].repeat(N, 1)  # children inherit s
        new_observed = self.unc_observed[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                   new_scaling, new_rotation, new_unc, new_observed)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        self.densification_postfix(self._xyz[selected_pts_mask],
                                   self._features_dc[selected_pts_mask],
                                   self._features_rest[selected_pts_mask],
                                   self._opacity[selected_pts_mask],
                                   self._scaling[selected_pts_mask],
                                   self._rotation[selected_pts_mask],
                                   self._unc[selected_pts_mask],
                                   self.unc_observed[selected_pts_mask])


# ---------------------------------------------------------------------------
# Confidence rendering (Eq 9), geometry detached, no CUDA changes
# ---------------------------------------------------------------------------
def render_confidence(viewpoint_camera, pc: GaussianModelWithUnc, pipe, scaling_modifier=1.0, eps=1e-3,
                      for_vis=False, probe=None):
    """Returns C_hat (H,W), accumulated opacity (H,W) and the probe map (H,W).

    Only pc._unc receives gradient: xyz, opacity, scale, rotation are detached
    and means2D is a fresh tensor, so densification statistics are untouched.
    for_vis=True renders never-supervised Gaussians with confidence c_min
    (u = u_max); use it only for evaluation, never for the loss.
    probe: optional (N,1) zeros tensor with requires_grad, rendered into channel 2.
    Its value is 0 everywhere, but d(sum of probe map over the mask)/d(probe_i) is
    the total blending weight of Gaussian i on supervised pixels. This marks which
    Gaussians are observed, independent of whether C_hat hits its clamp.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = RGBDGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach()
    means2D = torch.zeros_like(means3D)  # no grad, not the densification tensor
    opacity = pc.get_opacity.detach()
    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).detach()
    else:
        scales = pc.get_scaling.detach()
        rotations = pc.get_rotation.detach()

    gamma = pc.get_confidence  # (N,1), carries grad to _unc
    if for_vis:
        gamma = 1.0 / pc.get_uncertainty
    third = probe if probe is not None else torch.zeros_like(gamma)
    colors = torch.cat([gamma, torch.ones_like(gamma), third], dim=1)

    out, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    acc = out[1]  # sum_i a_i T_i = 1 - T_final
    c_hat = out[0] / acc.detach().clamp_min(eps)
    return c_hat, acc.detach(), out[2]


# ---------------------------------------------------------------------------
# VCURF target for one view (no grad), median-normalized
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_vcurf_target(view, gaussians, pipe, background, n_vcam=6, r_scale=0.1, beta=0.075,
                         min_vcam_views=2, min_fg_pixels=100, eps=1e-12):
    """Returns (D_tilde (H,W), valid (H,W) bool) or None.

    D per pixel = min over virtual cams of depth_l2 / median + min of rgb_l2 / median,
    following active/VC_sel.py.
    Supervision mask: a pixel is valid only if it is foreground and at least
    min_vcam_views virtual cams see it (warp succeeds). Other pixels (mostly image
    borders) carry no information: they are excluded from the loss, so no gradient
    flows from them, and Gaussians only seen there keep u = u_max.
    D_tilde = beta * D / median(D) so the optimal C_hat is 1 at the median pixel.
    """

    render_pkg = render(view, gaussians, pipe, background)
    depth = render_pkg['depth'].squeeze()  # (H,W)
    pred_img = render_pkg['render']        # (3,H,W)
    H, W = depth.shape
    fg = depth > 0
    if fg.sum() < min_fg_pixels:
        return None

    look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
    rd_c2w = rd_c2w.to(depth.device)
    d_median = depth[fg].median()
    K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                           fovX=view.FoVx, fovY=view.FoVy).to(depth.device)
    backwarp = BackwardWarping(out_hw=(H, W), device=depth.device, K=K)
    vcams = VirtualCam(view).get_N_near_cam_by_look_at(n_vcam, look_at=look_at, radiaus=r_scale * d_median)

    vir_depths, vir_imgs, rd2virs = [], [], []
    for vir_view in vcams:
        vpkg = render(vir_view, gaussians, pipe, background)
        vir_depths.append(vpkg['depth'].squeeze().unsqueeze(0))
        vir_imgs.append(vpkg['render'])
        rd2virs.append(vir_view.world_view_transform.transpose(0, 1) @ rd_c2w)
    vir_depths = torch.stack(vir_depths)  # (n,1,H,W)
    vir_imgs = torch.stack(vir_imgs)      # (n,3,H,W)
    rd2virs = torch.stack(rd2virs)

    rd_depths = depth[None, None].repeat(n_vcam, 1, 1, 1)
    rd_imgs = pred_img[None].repeat(n_vcam, 1, 1, 1)
    vir2rd_imgs, vir2rd_depths, nv_mask = backwarp(img_src=vir_imgs, depth_src=vir_depths,
                                                   depth_tgt=rd_depths, tgt2src_transform=rd2virs)
    invalid = nv_mask.bool()  # (n,1,H,W)

    depth_l2 = (vir2rd_depths - rd_depths) ** 2
    rgb_l2 = ((vir2rd_imgs - rd_imgs) ** 2).mean(1, keepdim=True)
    depth_l2 = depth_l2.masked_fill(invalid, float('inf'))
    rgb_l2 = rgb_l2.masked_fill(invalid, float('inf'))
    min_depth_l2 = depth_l2.min(0).values.squeeze(0)  # (H,W)
    min_rgb_l2 = rgb_l2.min(0).values.squeeze(0)

    n_seen = n_vcam - invalid.sum(0).squeeze(0)  # virtual cams that see each pixel
    valid = fg & (n_seen >= min_vcam_views)
    if valid.sum() < min_fg_pixels:
        return None

    d_term = min_depth_l2 / (min_depth_l2[valid].median() + eps)
    c_term = min_rgb_l2 / (min_rgb_l2[valid].median() + eps)
    D = d_term + c_term
    D = D / (D[valid].median() + eps)
    # clamp outliers so the fp16 cache cannot overflow; C_hat saturates at 1e-3 long before this
    D = torch.where(valid, D.clamp(max=1e4), torch.zeros_like(D))
    return beta * D, valid


class VcurfTargetCache:
    """Per-view cache of VCURF targets; recomputed when older than refresh_every iterations."""

    def __init__(self, refresh_every=500):
        self.refresh_every = refresh_every
        self.store = {}

    def clear(self):
        self.store = {}

    def get(self, view, iteration, compute_fn):
        key = view.uid
        entry = self.store.get(key)
        stale = entry is None or self.refresh_every <= 0 or (iteration - entry[2]) >= self.refresh_every
        if stale:
            tgt = compute_fn(view)
            if tgt is None:
                self.store.pop(key, None)
                return None
            D, valid = tgt
            self.store[key] = (D.half(), valid, iteration)
            entry = self.store[key]
        return entry[0].float(), entry[1]


def uncertainty_loss(c_hat, acc, D_tilde, valid, beta, c_min=1e-3, c_max=5.0, min_acc=0.5):
    """Eq 10 with VCURF as the error term, on supervised pixels only.

    Returns (loss, mask) or (None, mask) if no pixel is usable. Pixels outside the
    mask contribute nothing, so they send no gradient to any Gaussian.
    """
    mask = valid & (acc > min_acc)
    if mask.sum() == 0:
        return None, mask
    c = torch.clamp(c_hat, min=c_min, max=c_max)
    per_pixel = c * D_tilde - beta * torch.log(c)
    return per_pixel[mask].mean(), mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, uargs):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModelWithUnc(dataset.sh_degree, unc_lr=uargs.unc_lr)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cache = VcurfTargetCache(refresh_every=uargs.vcurf_refresh_every)
    compute_fn = lambda v: compute_vcurf_target(v, gaussians, pipe, background,
                                                n_vcam=uargs.n_vcam, r_scale=uargs.r_scale,
                                                beta=uargs.unc_beta, min_vcam_views=uargs.unc_min_vcam_views)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Current mapping frame
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Photometric loss (unchanged from train.py)
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Uncertainty loss: only updates gaussians._unc
        L_u = None
        in_reset_window = (iteration < opt.densify_until_iter and
                           iteration % opt.opacity_reset_interval <= uargs.unc_pause_after_reset)
        if iteration >= uargs.unc_from_iter and not in_reset_window:
            tgt = cache.get(viewpoint_cam, iteration, compute_fn)
            if tgt is not None:
                D_tilde, valid = tgt
                probe = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda", requires_grad=True)
                c_hat, acc, probe_map = render_confidence(viewpoint_cam, gaussians, pipe, probe=probe)
                L_u, sup_mask = uncertainty_loss(c_hat, acc, D_tilde, valid, beta=uargs.unc_beta,
                                                 c_min=gaussians.c_min, min_acc=uargs.unc_min_acc)
                if L_u is not None:
                    # probe term is exactly 0; it only exposes per-Gaussian weight on supervised pixels
                    loss = loss + uargs.unc_lambda * L_u + probe_map[sup_mask].sum()
                else:
                    probe = None

        loss.backward()
        if L_u is not None and probe is not None and probe.grad is not None:
            gaussians.mark_observed(probe.grad.squeeze(-1), uargs.unc_obs_weight)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, L_u, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (unchanged; uses only the RGB pass statistics)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, L_u, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if L_u is not None:
            tb_writer.add_scalar('uncertainty/L_u', L_u.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # rendered uncertainty 1/C_hat, scaled for display
                        c_hat, acc, _ = render_confidence(viewpoint, scene.gaussians, renderArgs[0], for_vis=True)
                        # log-scaled u in [0,1]: 0 -> u <= 1 (typical), 1 -> u_max (never supervised)
                        u_map = torch.log(1.0 / c_hat.clamp(1e-3, 5.0)) / math.log(scene.gaussians.u_max)
                        u_map = (u_map.clamp(0, 1) * (acc > 0.5)).clamp(0, 1)
                        tb_writer.add_images(config['name'] + "_view_{}/uncertainty".format(viewpoint.image_name), u_map[None, None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("uncertainty/log_u_histogram", -scene.gaussians._unc.detach(), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script with per-Gaussian uncertainty")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # uncertainty options
    parser.add_argument('--unc_lr', type=float, default=1e-3, help="learning rate of s (u = exp(-s))")
    parser.add_argument('--unc_beta', type=float, default=0.075, help="beta in Eq 10")
    parser.add_argument('--unc_lambda', type=float, default=1.0, help="weight of L_u (no effect under Adam, s only gets L_u)")
    parser.add_argument('--unc_from_iter', type=int, default=3000, help="start the VCURF loss at this iteration")
    parser.add_argument('--unc_pause_after_reset', type=int, default=200, help="skip L_u this many iters after each opacity reset")
    parser.add_argument('--unc_min_acc', type=float, default=0.5, help="ignore pixels with accumulated opacity below this")
    parser.add_argument('--vcurf_refresh_every', type=int, default=500, help="recompute a view's VCURF target after this many iters; 0 = every time")
    parser.add_argument('--n_vcam', type=int, default=6)
    parser.add_argument('--r_scale', type=float, default=0.1)
    parser.add_argument('--unc_obs_weight', type=float, default=1e-3,
                        help="a Gaussian counts as observed once its summed weight on supervised pixels exceeds this")
    parser.add_argument('--unc_min_vcam_views', type=int, default=2,
                        help="supervise a pixel only if at least this many virtual cams see it")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet, seed=args.seed)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    print("\nTraining complete.")
