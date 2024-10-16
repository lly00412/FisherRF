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
def render_uncertainty(view, gaussians, pipeline, background, hessian_color_C, hessian_color_D,args):
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
    render_pkg_D = modified_render(view, gaussians, pipeline, background, override_color=hessian_color_D)
    uncertanity_map_D = reduce(render_pkg["render"], "c h w -> h w", "mean")

    ###########################
    #  rendering vcams
    ###########################
    # TODO: change theta to be different values and show the difference on uncertainty estimation
    if args.render_vcam:

        # create sampling sphere by median depth of the scene center
        look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
        D_median = depth.clone().flatten().median(0).values
        radiaus = 0.1*D_median

        rd_c2w = rd_c2w.to(depth.device)
        K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                               fovX=view.FoVx, fovY=view.FoVy).to(depth.device)  # (4,4)
        GetVcam = VirtualCam(view)
        backwarp = BackwardWarping(out_hw=(view.image_height, view.image_width),
                                   device=depth.device, K=K)

        # random sampling n virtual camera at a sphere centering at real camera
        for N in args.n_vcam:
            rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
            rd_depths = rd_depth.repeat(N, 1, 1, 1)
            rd_pred_imgs = pred_img.clone().unsqueeze(0).repeat(N, 1, 1, 1)
            vir_depths = []
            vir_pred_imgs = []
            rd2virs = []
            Vcams = GetVcam.get_N_near_cam_by_look_at(N,look_at=look_at, radiaus=radiaus)
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
            rd2virs = torch.stack(rd2virs)
            vir2rd_pred_imgs, vir2rd_depths, nv_mask = backwarp(img_src=rd_pred_imgs, depth_src=vir_depths,
                                                                depth_tgt=rd_depths,
                                                                tgt2src_transform=rd2virs)
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
            rests[f'depth_l2({N} vcams)'] = depth_l2

            # rgb uncertainty
            vir2rd_pred_sum = vir2rd_pred_imgs.sum(0).mean(0, keepdim=True)
            rendering_ = pred_img.mean(0, keepdim=True)
            vir2rd_pred = torch.zeros_like(rendering_)
            vir2rd_pred[numels > 0] = vir2rd_pred_sum[numels > 0] / numels[numels > 0]
            rgb_l2 = (rendering_ - vir2rd_pred) ** 2
            rgb_l2 = rgb_l2.squeeze(0)
            rests[f'rgb_l2({N} vcams)'] = rgb_l2

    return pred_img, uncertanity_map_C, uncertanity_map_D, pixel_gaussian_counter, depth, rests

def render_set(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    render_path = os.path.join(model_path, "renders")
    eval_path = os.path.join(model_path, f"eval_seed_{args.seed}")
    depth_path = os.path.join(model_path, "depth")
    error_path = os.path.join(model_path, "error")
    roc_path = os.path.join(model_path, f"roc_seed_{args.seed}")

    makedirs(render_path, exist_ok=True)
    makedirs(eval_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(roc_path, exist_ok=True)

    if args is not None:
        if args.render_vcam:
            vir_path = os.path.join(model_path, "vcams")
            makedirs(vir_path, exist_ok=True)

    params = capture(gaussians)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    xyz = params[0]
    # filter_out_idx = [name2idx[k] for k in ["rotation", "rgb", "sh"]]
    filter_out_idx = [name2idx[k] for k in ["rotation", "scale", "xyz", "opacity"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device
    # H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)
    H_per_gaussian_C = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)
    H_per_gaussian_D = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)
    if not args.depth_only:
        # TODO: We can also use all the views, here the train views are just a subset of training cameras
        for idx, view in enumerate(tqdm(itertools.chain(train_views, test_views), desc="Rendering progress")):

            # rendering = render(view, gaussians, pipeline, background)["render"]

            render_pkg = modified_render(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            depth = render_pkg["depth"]
            pred_img.backward(gradient=torch.ones_like(pred_img),retain_graph=True)
            pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
            # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=torch.ones_like(params[1]))
            H_per_gaussian_C += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
            # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=H_per_gaussian.detach())
            optim.zero_grad(set_to_none=True)

            depth.backward(gradient=torch.ones_like(depth),retain_graph=True)
            H_per_gaussian_D += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
            optim.zero_grad(set_to_none=True)

            split = "train" if idx < len(train_views) else "test"

            torchvision.utils.save_image(pred_img.detach(), os.path.join(render_path, f"{split}_{view.image_name}.png"))
    else:
        H_per_gaussian_C += 1

    hessian_color_C = repeat(H_per_gaussian_C.detach(), "n -> n c", c=3)
    hessian_color_D = repeat(H_per_gaussian_D.detach(), "n -> n c", c=3)

    ROCs = {}
    AUCs = {}
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):
            
            to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
            pts3d_homo = to_homo(xyz)
            pts3d_cam = pts3d_homo @ view.world_view_transform
            gaussian_depths = pts3d_cam[:, 2, None]

            cur_hessian_color_C = hessian_color_C * gaussian_depths.clamp(min=0)
            cur_hessian_color_D = hessian_color_D * gaussian_depths.clamp(min=0)
            pred_img, uncertanity_map_C, uncertanity_map_D, pixel_gaussian_counter, depth, rests = render_uncertainty(view, gaussians, pipeline, background, cur_hessian_color_C, cur_hessian_color_D, args)

            ################################
            #  save all outputs
            ################################
            mask = (depth>0.)

            # save depth
            plt.figure(facecolor='white')
            sns.heatmap(depth.detach().cpu(), square=True,mask=~mask.detach().cpu().numpy())
            plt.savefig(os.path.join(depth_path, f"{view.image_name}.jpg"))
            plt.close()

            # save error
            plt.figure(facecolor='white')
            sns.heatmap(rests['rgb_err'].detach().cpu(), square=True,mask=~mask.detach().cpu().numpy())
            plt.savefig(os.path.join(error_path, f"{view.image_name}.jpg"))
            plt.close()

            if args.render_vcam:
                # save l2 diff
                for N in args.n_vcam:
                    plt.figure(facecolor='white')
                    sns.heatmap(rests[f'rgb_l2({N} vcams)'].detach().cpu(), square=True, mask=~mask.detach().cpu().numpy())
                    plt.savefig(os.path.join(eval_path, f"rgbl2_{N}_vcams_{view.image_name}.jpg"))
                    plt.close()

                    plt.figure(facecolor='white')
                    sns.heatmap(rests[f'depth_l2({N} vcams)'].detach().cpu(), square=True,mask=~mask.detach().cpu().numpy())
                    plt.savefig(os.path.join(eval_path, f"depthl2_{N}_vcams_{view.image_name}.jpg"))
                    plt.close()

            # save fisherRF
            sns.heatmap(torch.log(uncertanity_map_C / pixel_gaussian_counter).detach().cpu(), square=True)
            plt.savefig(os.path.join(eval_path, f"fisher_C_{view.image_name}.jpg"))
            plt.close()

            sns.heatmap(torch.log(uncertanity_map_D / pixel_gaussian_counter).detach().cpu(), square=True)
            plt.savefig(os.path.join(eval_path, f"fisher_D_{view.image_name}.jpg"))
            plt.close()

            # save raw output
            save_rests = {}
            save_rests['fisher_D'] = uncertanity_map_D.cpu()
            for k in rests.keys():
                save_rests[k] = rests[k].cpu()
            np.savez(os.path.join(eval_path, f"uncertainty_{idx:03d}_{view.image_name}.npz"),
                     uncertanity_map=uncertanity_map_C.cpu(), pixel_gaussian_counter=pixel_gaussian_counter.cpu(),
                     depth=depth.cpu(), rests=save_rests,
                     )

            ################################
            #  compute auc
            ################################
            opt_label = 'rgb_err'
            values = {
                'fisherRF_C':uncertanity_map_C[mask].flatten(),
                'fisherRF_D': uncertanity_map_D[mask].flatten(),
            }

            for k in rests.keys():
                values[k] = rests[k][mask].flatten()

            rocs = {}
            aucs = {}
            for val in values.keys():
                roc, auc = compute_roc(opt=values[opt_label], est=values[val], intervals=20)
                rocs[val] = np.array(roc)
                aucs[val] = auc
                if val not in ROCs.keys():
                    ROCs[val] = [roc]
                    AUCs[val] = [auc]
                else:
                    ROCs[val].append(roc)
                    AUCs[val].append(auc)

            plot_file = os.path.join(roc_path, '{0:05d}'.format(idx) + ".jpg")
            txt_file = os.path.join(roc_path, '{0:05d}'.format(idx) + ".txt")
            plot_roc(ROC_dict=rocs, fig_name=plot_file, opt_label=opt_label,intervals=20)
            write_auc(AUC_dict=aucs, txt_name=txt_file)


        for val in ROCs.keys():
            ROCs[val] = np.array(ROCs[val]).mean(0)
            AUCs[val] = np.array(AUCs[val]).mean(0)
        summary_plot = os.path.join(roc_path, f'{name}' + ".png")
        summary_txt = os.path.join(roc_path, f'{name}' + ".txt")
        plot_roc(ROC_dict=ROCs, fig_name=summary_plot, opt_label=opt_label,intervals=20)
        write_auc(AUC_dict=AUCs, txt_name=summary_txt)

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
            plt.clf()

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
    parser.add_argument("--render_vcam", action="store_true", help="render uncertainty from virtual cameras")
    parser.add_argument("--n_vcam", nargs="+", default=[2,4,6,8], type=int, help="num of virtual cameras")
    # parser.add_argument("--thetas", nargs="+", type=float, default=[1,3,5,7],help="angle of turning virtual cameras")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet,seed=args.seed)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)
