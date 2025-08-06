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

        candidate_cameras = scene.getCandidateCameras()
        # TODO: To be change latter
        depth_scores = []
        color_scores = []
        occ_scores = []
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
            numels[numels==0] = 1.0

            ###############################
            #  compute uncertainty by l2 diff
            ################################
            if bg_mask.float().sum()<(h*w):  # something can be rendered
                # weight scores
                confs = 1.0 - nv_mask.sum(0)+1.0/(self.n_vcam+1.0)

                depth_l2 = (vir2rd_depths - rd_depths) **2
                avg_depth_l2 = torch.squeeze(depth_l2.sum(0) / numels)
                weight_depth_l2 = avg_depth_l2*confs.squeeze()
                depth_scores.append(weight_depth_l2[~bg_mask].mean().item())

                # rgb uncertainty
                rgb_l2 = ((vir2rd_pred_imgs - rd_pred_imgs) ** 2).mean(1)
                avg_rgb_l2 = torch.squeeze(rgb_l2.sum(0) / numels)
                weight_rgb_l2 = avg_rgb_l2 * confs.squeeze()
                color_scores.append(weight_rgb_l2[~bg_mask].mean().item())

                # count vaild pixels
                nv_pixels = nv_mask.sum(0).squeeze()+bg_mask.float()
                occ_mask = (nv_pixels > 0)
                total_pixels = bg_mask.numel()
                occ_weight = occ_mask.float().sum() /total_pixels
                occ_scores.append(occ_weight.item())
                # vcurf_scores.append((vc_scores[~bg_mask].mean() * weight).item())
                # del norm_rgb_sigmas, norm_depth_sigmas, bg_mask
                del avg_depth_l2, avg_rgb_l2, bg_mask
            else:
                depth_scores.append(1.0)
                color_scores.append(1.0)
                occ_scores.append(1.0)

        # vcurf_scores = np.array(vcurf_scores)
        occ_scores = np.array(occ_scores)
        v_mask = (occ_scores < 1.0)

        depth_scores = np.array(depth_scores)
        v_depth = depth_scores[v_mask]
        exp_depth_scores = np.exp(v_depth - np.max(v_depth))  # for numerical stability
        softmax_depth_scores = exp_depth_scores / np.sum(exp_depth_scores)
        v_depth_scores = np.ones_like(occ_scores)
        v_depth_scores[v_mask] = softmax_depth_scores

        color_scores = np.array(color_scores)
        v_color = color_scores[v_mask]
        exp_color_scores = np.exp(v_color - np.max(v_color))  # for numerical stability
        softmax_color_scores = exp_color_scores / np.sum(exp_color_scores)
        v_color_scores = np.ones_like(occ_scores)
        v_color_scores[v_mask] = softmax_color_scores

        vcurf_scores = occ_scores*v_depth_scores*v_color_scores

        selected_idxs = np.argsort(vcurf_scores)[-num_views:]
        selected_view_idx = [candidate_views[k] for k in selected_idxs]
        return selected_view_idx
    
    def forward(self, x):
        return x
