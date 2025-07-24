import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import network_gui
from scene import Scene
import math
import os,sys

from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import BackwardWarping, extract_scene_center_and_C2W
from scene.cameras import VirtualCam
import torch.nn as nn

from gaussian_renderer import modified_render


class VCSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.n_vcam = args.n_vcam
        self.scale =  args.r_scale
    
    @torch.no_grad()
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        # off load to cpu to avoid oom with greedy algo
        # device = params[0].device if num_views == 1 else "cpu"
        device = "cpu" # we have to load to cpu because of inflation
        candidate_cameras = scene.getCandidateCameras()
        # TODO: To be change latter
        vcurf_scores = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating Virtual Camera Uncertainty on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            depth = render_pkg['depth']
            pred_img = render_pkg["render"]

            look_at, rd_c2w = extract_scene_center_and_C2W(depth, cam)
            D_median = depth.clone().flatten().median(0).values

            rd_c2w = rd_c2w.to(depth.device)
            K = getIntrinsicMatrix(width=cam.image_width, height=cam.image_height,
                                   fovX=cam.FoVx, fovY=cam.FoVy).to(depth.device)  # (4,4)
            GetVcam = VirtualCam(cam)
            backwarp = BackwardWarping(out_hw=(cam.image_height, cam.image_width),
                                       device=depth.device, K=K)


            radiaus = self.scale * D_median
            Vcams = GetVcam.get_N_near_cam_by_look_at(self.n_vcam, look_at=look_at, radiaus=radiaus)

            rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
            rd_depths = rd_depth.repeat(self.n_vcam, 1, 1, 1)
            rd_pred_imgs = pred_img.clone().unsqueeze(0).repeat(self.n_vcam, 1, 1, 1)

            vir_depths = []
            vir_pred_imgs = []
            rd2virs = []
            for vir_view in Vcams:
                vir_render_pkg = modified_render(vir_view, gaussians, pipe, background)
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
            ###############################
            #  fillter out backgroud pixels and occlusion mask
            ###############################
            # bg_mask_per_channel = (pred_img == background.view(3, 1, 1))
            # bg_mask = bg_mask_per_channel.all(dim=0)
            bg_mask = ~(depth.squeeze() > 0)
            _, h,w = pred_img.shape

            numels = float(self.n_vcam) - nv_mask.sum(0)
            numels[numels==0] == 1.0

            ###############################
            #  compute uncertainty by l2 diff
            ################################
            # depth uncertainty -- if find the avg
            depth_l2 = (vir2rd_depths - rd_depths) **2
            avg_depth_l2 = torch.squeeze(depth_l2.sum(0) / numels)

            # min_depth_l2 = depth_l2.min(0).values.squeeze()

            MIN_VALUE = avg_depth_l2[~bg_mask].flatten().min()
            MAX_VALUE = avg_depth_l2[~bg_mask].flatten().max()
            norm_depth_sigmas = torch.zeros_like(avg_depth_l2)
            norm_depth_sigmas[~bg_mask] = (avg_depth_l2[~bg_mask] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

            # rgb uncertainty
            rgb_l2 = ((vir2rd_pred_imgs - rd_pred_imgs) ** 2).mean(1)
            avg_rgb_l2 = torch.squeeze(rgb_l2.sum(0) / numels)
            MIN_VALUE = avg_rgb_l2[~bg_mask].flatten().min()
            MAX_VALUE = avg_rgb_l2[~bg_mask].flatten().max()
            norm_rgb_sigmas = torch.zeros_like(avg_rgb_l2)
            norm_rgb_sigmas[~bg_mask] = (avg_rgb_l2[~bg_mask] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

            vc_scores = norm_rgb_sigmas + norm_depth_sigmas
            bg_pixels = bg_mask.float().sum()
            total_pixels = bg_mask.numel()
            weight = bg_pixels / total_pixels
            vcurf_scores.append((vc_scores[~bg_mask].mean() * weight).item())

        vcurf_scores = np.array(vcurf_scores)
        selected_idxs = np.argsort(vcurf_scores)[-num_views:]
        selected_view_idx = [candidate_views[k] for k in selected_idxs]
        return selected_view_idx
    
    def forward(self, x):
        return x
