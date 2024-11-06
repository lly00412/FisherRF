#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.camera_utils import rand_rotation_matrix
from scene.cameras import Camera
from gaussian_renderer import modified_render
from einops import reduce, repeat, rearrange
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from active.schema import schema_dict, override_test_idxs_dict, override_train_idxs_dict

##### Virtual cameras
from scene.cameras import VirtualCam
from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import *
from utils.uncert_utils import *

def capture(self):
    return (
        self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
        # self.optimizer.state_dict(),
        # self.spatial_lr_scale,
    )

@torch.no_grad()
def render_uncertainty_for_emsemble(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    gaussians = GaussianModel(dataset.sh_degree)
    # use every frames
    if hasattr(args, 'override_idxs'):
        override_train_idxs = list(range(10_000))
        override_test_idxs = override_test_idxs_dict[args.override_idxs]
    else:
        override_train_idxs = None
        override_test_idxs = None
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, override_train_idxs=override_train_idxs,
                  override_test_idxs=override_test_idxs)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_path = os.path.join(args.model_path, "emsemble_renders")
    roc_path = os.path.join(args.model_path, "emsemble_roc")
    makedirs(render_path, exist_ok=True)
    makedirs(roc_path, exist_ok=True)

    test_views = scene.getTestCameras()
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):
            gt_img = view.original_image[0:3, :, :]
            depths = []
            pred_imgs = []
            for seed in args.emsemble_seeds:
                render_path = os.path.join(args.model_path,seed, "renders")
                render_file = os.path.join(render_path,f"emsemble_{idx:03d}_{view.image_name}.npz")
                emsemble_outputs = np.load(file=render_file,allow_pickle=True)
                depths.append(emsemble_outputs['depth'])
                pred_imgs.append(emsemble_outputs['pred_img'])

    ###########################
    #  rendering RGB, depth & error
    ###########################
    rests = {}
    render_pkg = modified_render(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    # pred_img.backward(gradient=torch.ones_like(pred_img))
    gt_img = view.original_image[0:3, :, :]
    pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
    rgb_err = torch.mean((pred_img - gt_img)**2,0)
    rests['rgb_err'] = rgb_err

    # compute H by render RGB
    render_pkg = modified_render(view, gaussians, pipeline, background, override_color=hessian_color_C)
    depth = render_pkg["depth"]
    uncertanity_map_C = reduce(render_pkg["render"], "c h w -> h w", "mean")

    # compute H by render Depth
    # render_pkg_D = modified_render(view, gaussians, pipeline, background, override_color=hessian_color_D)
    # uncertanity_map_D = reduce(render_pkg["render"], "c h w -> h w", "mean")

    ###########################
    #  rendering vcams
    ###########################
    # TODO: change theta to be different values and show the difference on uncertainty estimation
    if args.render_vcam:

        # create sampling sphere by median depth of the scene center
        look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
        D_median = depth.clone().flatten().median(0).values
        # radiaus = 0.1*D_median

        rd_c2w = rd_c2w.to(depth.device)
        K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                               fovX=view.FoVx, fovY=view.FoVy).to(depth.device)  # (4,4)
        GetVcam = VirtualCam(view)
        backwarp = BackwardWarping(out_hw=(view.image_height, view.image_width),
                                   device=depth.device, K=K)

        # random sampling n virtual camera at a sphere centering at real camera
        for N in args.n_vcam:
            for scale in args.r_scale:
                radiaus = scale*D_median
                Vcams = GetVcam.get_N_near_cam_by_look_at(N, look_at=look_at, radiaus=radiaus)

                rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
                rd_depths = rd_depth.repeat(N, 1, 1, 1)
                rd_pred_imgs = pred_img.clone().unsqueeze(0).repeat(N, 1, 1, 1)

                vir_depths = []
                vir_pred_imgs = []
                rd2virs = []
                for vir_view in Vcams:
                    vir_render_pkg = modified_render(vir_view, gaussians, pipeline, background)
                    vir_depth = vir_render_pkg['depth']
                    vir_pred_img = vir_render_pkg['render']
                    vir_w2c = vir_view.world_view_transform.transpose(0, 1)
                    rd2vir = vir_w2c @ rd_c2w
                    rd2virs.append(rd2vir)
                    vir_depths.append(vir_depth.unsqueeze(0))
                    vir_pred_imgs.append(vir_pred_img)
                vir_depths = torch.stack(vir_depths)
                vir_pred_imgs = torch.stack(vir_pred_imgs)
                rd2virs = torch.stack(rd2virs)
                vir2rd_pred_imgs, vir2rd_depths, nv_mask = backwarp(img_src=vir_pred_imgs, depth_src=vir_depths,
                                                                    depth_tgt=rd_depths,
                                                                    tgt2src_transform=rd2virs)
                breakpoint()
                torchvision.utils.save_image(pred_img.detach(),"./output/m360/debug/rgb_pred.jpg")
                torchvision.utils.save_image(vir_pred_imgs[0].detach(), "./output/m360/debug/vir_rgb_pred_0.jpg")
                torchvision.utils.save_image(vir2rd_pred_imgs[0].detach(), "./output/m360/debug/vir2rd_rgb_pred_0.jpg")
                ################################
                #  compute uncertainty by l2 diff
                ################################
                # depth uncertainty
                vir2rd_depth_sum = vir2rd_depths.sum(0)
                numels = float(N) - nv_mask.sum(0)
                vir2rd_depth = torch.zeros_like(rd_depth.squeeze(0))
                vir2rd_depth[numels > 0] = vir2rd_depth_sum[numels > 0] / numels[numels > 0]
                depth_l2 = (rd_depth.squeeze(0) - vir2rd_depth) ** 2
                depth_l2 = depth_l2.squeeze(0)
                MIN_VALUE = depth_l2.flatten().min()
                MAX_VALUE = depth_l2.flatten().max()
                norm_depth_sigmas = (depth_l2 - MIN_VALUE) / (MAX_VALUE- MIN_VALUE)
                # rests[f'depth_l2({N} vcams, {scale} med)'] = depth_l2

                # rgb uncertainty
                vir2rd_pred_sum = vir2rd_pred_imgs.sum(0).mean(0, keepdim=True)
                rendering_ = pred_img.mean(0, keepdim=True)
                vir2rd_pred = torch.zeros_like(rendering_)
                vir2rd_pred[numels > 0] = vir2rd_pred_sum[numels > 0] / numels[numels > 0]
                rgb_l2 = (rendering_ - vir2rd_pred) ** 2
                rgb_l2 = rgb_l2.squeeze(0)
                MIN_VALUE = rgb_l2.flatten().min()
                MAX_VALUE = rgb_l2.flatten().max()
                norm_rgb_sigmas = (rgb_l2 - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
                vcu = norm_rgb_sigmas + norm_depth_sigmas
                rests[f'vcu({N} vcams, {scale} med)'] = vcu

    return pred_img, uncertanity_map_C, pixel_gaussian_counter, depth, rests

def render_set(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    render_path = os.path.join(model_path, "renders")
    makedirs(render_path, exist_ok=True)

    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):

            render_pkg = modified_render(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            depth = render_pkg["depth"]
            np.savez(os.path.join(render_path, f"emsemble_{idx:03d}_{view.image_name}.npz"),
                     depth=depth.cpu(), pred_img=pred_img.cpu(),
                     )

def render_set_current(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    eval_path = os.path.join(model_path, "eval")

    makedirs(eval_path, exist_ok=True)

    params = capture(gaussians)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    filter_out_idx = [name2idx[k] for k in ["rotation"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device

    for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):

        render_pkg = modified_render(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        pred_img.backward(gradient=torch.ones_like(pred_img))
        pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
        H_per_gaussian = sum(reduce(p.grad.detach(), "n ... -> n", "sum") for p in params)

        with torch.no_grad():
            hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)

            # compute depth of gaussian in current view
            to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
            pts3d_homo = to_homo(params[0])
            pts3d_cam = pts3d_homo @ view.world_view_transform
            gaussian_depths = pts3d_cam[:, 2, None]

            hessian_color = hessian_color * gaussian_depths

            render_pkg = modified_render(view, gaussians, pipeline, background, override_color=hessian_color)

            uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean")
            depth = render_pkg["depth"]

            # sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).clamp(min=0).detach().cpu(), square=True)
            # plt.savefig(f"./uncern.jpg")
            # plt.savefig(f"./uncern_all.jpg")
            # plt.clf()

            torchvision.utils.save_image(pred_img.detach(), os.path.join(eval_path, f"render_{view.image_name}.png"))
            sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).clamp(min=0).detach().cpu(), square=True)
            plt.savefig(os.path.join(eval_path, f"heatmap_{view.image_name}.jpg"))
            plt.clf()

            np.savez(os.path.join(eval_path, f"uncertainty_{idx:03d}_{view.image_name}.npz"), 
                        uncertanity_map=uncertanity_map.cpu(), pixel_gaussian_counter=pixel_gaussian_counter.cpu(),
                        depth=depth.cpu(),
                        )

            optim.zero_grad(set_to_none = True) 


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    gaussians = GaussianModel(dataset.sh_degree)

    # override_train_idxs = override_train_idxs_dict.get(args.override_idxs, None)
    # use every frames
    if hasattr(args, 'override_idxs'):
        override_train_idxs = list(range(10_000))
        override_test_idxs = override_test_idxs_dict[args.override_idxs]
    else:
        override_train_idxs = None
        override_test_idxs = None
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, override_train_idxs=override_train_idxs, override_test_idxs=override_test_idxs)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.current:
        render_set_current(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene.getTestCameras(), gaussians, pipeline, background, camera_extent=scene.cameras_extent, args=args)
    else:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene.getTestCameras(), gaussians, pipeline, background, camera_extent=scene.cameras_extent, args=args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--perturb_scale", default=1., type=float)
    parser.add_argument("--inflate_factor", default=5, type=int)
    parser.add_argument("--override_idxs", default=None, type=str, help="speical test idxs on uncertainty evaluation")
    parser.add_argument("--depth_only", action="store_true", help="render depth only")
    parser.add_argument("--current", action="store_true", help="render uncertainty from current view")
    parser.add_argument("--emsemble_seeds", nargs="+", default=[0,500,1000,2000,600], type=int, help="seeds for emsemble models")
    # parser.add_argument("--thetas", nargs="+", type=float, default=[1,3,5,7],help="angle of turning virtual cameras")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    emsemble_path = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet,seed=args.seed)

    for seed in args.emsemble_seeds:
        args.model_path = os.path.join(emsemble_path,seed)
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)

    render_uncertainty_for_emsemble(model.extract(args), args.iteration, pipeline.extract(args), args):

    # TODO: after collecting all the seeds, need to compute the variance of all settings