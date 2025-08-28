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
from binary_classifer_on_view_selection import BinarryClassifier,load_ckpt

def get_central_moments(U):
    x = U.flatten().float()
    mu = x.mean()
    var = x.var(unbiased=False)
    sigma = x.std(unbiased=False)
    # add small eps for numerical stability if needed
    eps = 1e-8
    z = (x - mu) / (sigma + eps)

    skewness = (z ** 3).mean()  # γ1
    kurtosis = (z ** 4).mean()  # γ2 (Pearson, not excess)
    excess_kurtosis = kurtosis - 3
    moments = torch.tensor([mu, var, skewness, excess_kurtosis])
    return moments

def log_histogram(uncertainty_map, bins=10, eps=1e-8):
    # Flatten to 1D
    values = uncertainty_map.flatten()

    # Avoid log(0) by adding eps
    min_val = values.min().clamp(min=eps)
    max_val = values.max().clamp(min=eps)

    # Compute log-spaced bin edges
    bin_edges = torch.logspace(min_val.log10().item(), max_val.log10().item(), steps=bins + 1, device=values.device)

    # Assign each value to a bin index
    bin_indices = torch.bucketize(values, bin_edges, right=False)

    # Count occurrences (drop last bin edge to match histogram shape)
    hist = torch.bincount(bin_indices, minlength=bins + 1)[:bins].float()

    return hist, bin_edges


class MLPSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.n_vcam = args.n_vcam
        self.scale =  args.r_scale
        self.mlp = BinarryClassifier(indim=25*3, n_classes=2)
        load_ckpt(self.mlp,args.ckpt_path)
        self.mlp.eval()

    @torch.no_grad()
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidated_idxs, candidate_moments, depth_hists, color_hists, nv_pixels = self.generate_features(gaussians, Scene, num_views, pipe, background, exit_func)

        selected_idxs = np.argsort(vcurf_scores)[-num_views:]
        selected_view_idx = [candidate_views[k] for k in selected_idxs]
        return selected_view_idx


    def generate_features(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))
        candidate_cameras = scene.getCandidateCameras()

        depth_moments = []
        color_moments = []
        depth_hists = []
        color_hists = []
        nv_pixels = []
        candidated_idxs = []
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
            if bg_mask.float().sum()<(0.5*h*w):  # something can be rendered
                depth_l2 = (vir2rd_depths - rd_depths) **2
                avg_depth_l2 = torch.squeeze(depth_l2.sum(0) / numels)
                depth_moment = get_central_moments(avg_depth_l2[~bg_mask])
                depth_moments.append(depth_moment)

                d_hist, _ = log_histogram(avg_depth_l2[~bg_mask], bins=10, eps=1e-8)
                depth_hists.append(d_hist)

                # rgb uncertainty
                rgb_l2 = ((vir2rd_pred_imgs - rd_pred_imgs) ** 2).mean(1)
                avg_rgb_l2 = torch.squeeze(rgb_l2.sum(0) / numels)
                color_moment = get_central_moments(avg_rgb_l2[~bg_mask])
                color_moments.append(color_moment)

                c_hist, _ = log_histogram(avg_rgb_l2[~bg_mask], bins=10, eps=1e-8)
                color_hists.append(c_hist)

                # occ pixels
                nv_pixels.append(nv_mask.sum().item())

                ## candidate idxs
                candidated_idxs.append(candidate_views[idx])

        depth_moments = torch.stack(depth_moments, dim=0)
        color_moments = torch.stack(color_moments, dim=0)
        candidate_moments = torch.cat([depth_moments, color_moments], dim=1)
        depth_hists = torch.stack(depth_hists, dim=0)
        color_hists = torch.stack(color_hists, dim=0)

        return candidated_idxs, candidate_moments, depth_hists, color_hists, nv_pixels

    @torch.no_grad()
    def pick_best_candidate(
            model: nn.Module,
            feats: torch.Tensor,  # shape: (N, 25)
            batch_size: int = 2048,
            device: torch.device | str | None = None,
            positive_class: int = 1,  # which class index means "j better than i"
            method: str = "tournament",  # "tournament" or "sequential"
            return_stats: bool = False,
    ):
        """
        Returns:
            best_idx (int)
            (optional) dict with 'win_counts' (N,), 'pairwise_matrix' (N,N) if method='tournament'
        """
        assert feats.ndim == 2 and feats.size(1) == 25, "feats must be (N,25)"
        N = feats.size(0)
        if N == 1:
            return (0, {"win_counts": torch.tensor([0]), "pairwise_matrix": torch.zeros(1, 1)}) if return_stats else 0

        model_was_training = model.training
        model.eval()

        dev = device if device is not None else next(model.parameters()).device
        feats = feats.to(dev)

        def _build_inputs(i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
            """Create [fi, fj, fi-fj] for matching index vectors (same shape)."""
            fi = feats[i_idx]  # (...,25)
            fj = feats[j_idx]  # (...,25)
            diff = fi - fj  # (...,25)
            return torch.cat([fi, fj, diff], dim=-1)  # (...,75)

        def _predict_better(i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
            """
            Returns probability that j is better than i (shape = i_idx.shape).
            """
            x = _build_inputs(i_idx, j_idx)
            logits = model(x)  # (...,2)
            if logits.ndim == 1:  # fallback if model returns a single logit
                # interpret positive logit as "j better"; map to prob via sigmoid
                return torch.sigmoid(logits)
            probs = torch.softmax(logits, dim=-1)
            return probs[..., positive_class]  # P(j better than i)

        if method == "sequential":
            # O(N): running champion
            best = 0
            for j in range(1, N):
                p = _predict_better(torch.tensor([best], device=dev), torch.tensor([j], device=dev))[0]
                if p > 0.5:
                    best = j
            if return_stats:
                # Optionally compute light stats: wins from champion pass (not full matrix)
                stats = {"win_counts": None, "pairwise_matrix": None}
                return best, stats
            return best

        elif method == "tournament":
            # O(N^2) in batches: compute all pairwise decisions i vs j (i!=j)
            win_counts = torch.zeros(N, device=dev, dtype=torch.int32)
            pairwise = torch.zeros((N, N), device=dev, dtype=torch.float32)

            # Build all (i,j) pairs with i != j
            ii, jj = torch.meshgrid(torch.arange(N, device=dev), torch.arange(N, device=dev), indexing='ij')
            mask = ii != jj
            i_flat = ii[mask].reshape(-1)
            j_flat = jj[mask].reshape(-1)

            # Batched inference
            for start in range(0, i_flat.numel(), batch_size):
                end = min(start + batch_size, i_flat.numel())
                p = _predict_better(i_flat[start:end], j_flat[start:end])  # P(j better than i)
                # A "win" for j if p > 0.5
                winners_j = p > 0.5

                # Update pairwise matrix: pairwise[i,j] = P(j > i)
                pairwise[i_flat[start:end], j_flat[start:end]] = p

                # Count wins efficiently
                # For entries where j wins, increment win_counts[j]
                if winners_j.any():
                    idx_winners = j_flat[start:end][winners_j]
                    win_counts.index_add_(0, idx_winners, torch.ones_like(idx_winners, dtype=win_counts.dtype))

            # Pick the index with maximum wins (ties -> lowest index)
            best_idx = int(torch.argmax(win_counts).item())

            if return_stats:
                stats = {
                    "win_counts": win_counts.detach().cpu(),
                    "pairwise_matrix": pairwise.detach().cpu(),  # NxN, pairwise[i,j]=P(j > i)
                }
                return best_idx, stats
            return best_idx

    def forward(self, x):
        return x
