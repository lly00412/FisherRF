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
from gaussian_renderer import render
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
from utils.vcam_utils import *

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
def render_uncertainty(view, gaussians, pipeline, background):
    ###########################
    #  rendering RGB, depth & error
    ###########################
    rests = {}
    render_pkg = render(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    # pred_img.backward(gradient=torch.ones_like(pred_img))
    gt_img = view.original_image[0:3, :, :]
    rgb_err = torch.mean((pred_img - gt_img)**2,0)
    rests['rgb_err'] = rgb_err
    uncertainty = render_pkg["variance"]  # (h , w)

    return pred_img, uncertainty, rests

def render_set(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    render_path = os.path.join(model_path, "renders")
    eval_path = os.path.join(model_path, f"eval")
    error_path = os.path.join(model_path, "error")
    roc_path = os.path.join(model_path, f"roc")

    makedirs(render_path, exist_ok=True)
    makedirs(eval_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(roc_path, exist_ok=True)

    ROCs = {}
    AUCs = {}
    AUSEs = {}
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):

            pred_img, uncertainty, rests = render_uncertainty(view, gaussians, pipeline, background)

            ################################
            #  save all outputs
            ################################
            # save outputs
            torchvision.utils.save_image(pred_img.detach(), os.path.join(render_path, f"{view.image_name}.png"))


            # # save depth
            # plt.figure(facecolor='white')
            # sns.heatmap(depth.detach().cpu(), square=True,mask=~mask.detach().cpu().numpy())
            # plt.savefig(os.path.join(depth_path, f"{view.image_name}.jpg"))
            # plt.close()

            # save error
            plt.figure(facecolor='white')
            sns.heatmap(rests['rgb_err'].detach().cpu(), square=True)
            plt.savefig(os.path.join(error_path, f"{view.image_name}.jpg"))
            plt.close()

            # if args.render_vcam:
            #     # save l2 diff
            #     for N in args.n_vcam:
            #         for scale in args.r_scale:
            #             plt.figure(facecolor='white')
            #             sns.heatmap(torch.log(rests[f'vcu({N} vcams, {scale} med)']).detach().cpu(), square=True, mask=~mask.detach().cpu().numpy())
            #             plt.savefig(os.path.join(eval_path, f"vcurf_{N}_vcams_{scale}_med_{view.image_name}.jpg"))
            #             plt.close()

            # save uncertainty
            sns.heatmap(torch.log(uncertainty).detach().cpu(), square=True)
            plt.savefig(os.path.join(eval_path, f"fisher_C_{view.image_name}.jpg"))
            plt.close()

            # sns.heatmap(torch.log(uncertanity_map_D / pixel_gaussian_counter).detach().cpu(), square=True)
            # plt.savefig(os.path.join(eval_path, f"fisher_D_{view.image_name}.jpg"))
            # plt.close()

            # save raw output
            save_rests = {}
            # save_rests['fisher_D'] = uncertanity_map_D.cpu()
            for k in rests.keys():
                save_rests[k] = rests[k].cpu()
            np.savez(os.path.join(eval_path, f"uncertainty_{idx:03d}_{view.image_name}.npz"),
                     uncertanity_map=uncertainty.cpu(), rests=save_rests,
                     )

            ################################
            #  compute auc
            ################################
            opt_label = 'rgb_err'
            values = {
                'activenerf':uncertainty.flatten(),
            }

            for k in rests.keys():
                values[k] = rests[k].flatten()

            rocs = {}
            aucs = {}
            auses = {}
            for val in values.keys():
                roc, auc = compute_roc(opt=values[opt_label], est=values[val], intervals=20)
                _, ause = compute_ause(opt=values[opt_label], est=values[val], intervals=100)
                rocs[val] = np.array(roc)
                aucs[val] = auc
                auses[val] = ause
                if val not in ROCs.keys():
                    ROCs[val] = [roc]
                    AUCs[val] = [auc]
                    AUSEs[val] = [ause]
                else:
                    ROCs[val].append(roc)
                    AUCs[val].append(auc)
                    AUSEs[val].append(ause)

            plot_file = os.path.join(roc_path, '{0:05d}'.format(idx) + ".jpg")
            auc_file = os.path.join(roc_path, '{0:05d}'.format(idx) + "auc.txt")
            ause_file = os.path.join(roc_path, '{0:05d}'.format(idx) + "ause.txt")
            plot_roc(ROC_dict=rocs, fig_name=plot_file, opt_label=opt_label,intervals=20)
            write_auc(AUC_dict=aucs, txt_name=auc_file)
            write_auc(AUC_dict=auses,txt_name=ause_file)


        for val in ROCs.keys():
            ROCs[val] = np.array(ROCs[val]).mean(0)
            AUCs[val] = np.array(AUCs[val]).mean(0)
            AUSEs[val] = np.array(AUSEs[val]).mean(0)
        summary_plot = os.path.join(roc_path, f'{name}' + ".png")
        summary_auc = os.path.join(roc_path, f'{name}' + ".txt")
        summary_ause = os.path.join(roc_path, f'{name}' + "ause.txt")
        plot_roc(ROC_dict=ROCs, fig_name=summary_plot, opt_label=opt_label,intervals=20)
        write_auc(AUC_dict=AUCs, txt_name=summary_auc)
        write_auc(AUC_dict=AUSEs, txt_name=summary_ause)


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

        render_pkg = render(view, gaussians, pipeline, background)
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

            render_pkg = render(view, gaussians, pipeline, background, override_color=hessian_color)

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
    parser.add_argument("--render_vcam", action="store_true", help="render uncertainty from virtual cameras")
    parser.add_argument("--n_vcam", nargs="+", default=[2,4,6,8], type=int, help="num of virtual cameras")
    parser.add_argument("--r_scale", nargs="+", default=[0.05, 0.1, 0.25, 0.5], type=float, help="radiaus scale of the sampling space")
    # parser.add_argument("--thetas", nargs="+", type=float, default=[1,3,5,7],help="angle of turning virtual cameras")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet,seed=args.seed)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)


    # TODO: after collecting all the seeds, need to compute the variance of all settings