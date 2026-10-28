#
# Evaluate the per-Gaussian uncertainty u (trained by train_w_uncertainty.py)
# against VCURF on the test set.
#
# For every test view:
#   - render RGB, depth and the per-pixel RGB error (oracle) and the VCURF map with
#     the same code and formula as render_uncertainty_w_Vcams_v2.py (VCURF = mean over
#     valid virtual cams of depth / rgb l2 diff, each min-max normalized, then summed)
#   - ours: u map = 1 / C_hat, rendered from the per-Gaussian u (never-supervised
#     Gaussians count as u_max)
#   - AUC (compute_roc) and AUSE (compute_ause) of each method, with rgb_err as the
#     oracle, on foreground pixels (depth > 0), same as the existing scripts
#   - extra: AUSE of ours using VCURF as the oracle (how well u reproduces VCURF)
#   - depth error: T&T / Deep Blending have no GT depth, so the COLMAP sparse points
#     (<source_path>/sparse/0) projected into each view are the reference depth.
#     depth_err = |rendered depth / acc - COLMAP depth| at those pixels, and AUC / AUSE
#     of each method are computed with depth_err as the oracle (files *_depth*).
# Output layout follows render_uncertainty_w_Vcams_v2.py, under <model_path>/<save_dir>/:
#   renders/  depth/  error/  eval_seed_<s>/  roc_seed_<s>/
#
# Test views:
#   --split test      the llffhold test views of the training set (model trained with --eval)
#   --split nvs_test  a separate test set in the same COLMAP frame (e.g. TnT GT_data/<scene>/nvs_test):
#                     cameras from <test_source>/sparse/0, images from <test_source>/images, only the views in
#                     the "test" list of <test_source>/split_sparse_20.json, and dense GT depth from
#                     <test_source>/gt_depth/<view>.npy (made by gt_depthmaps_standalone.py, COLMAP frame, NaN = none)
#   --split train     quick check on training views
#
# Examples:
#   python render_uncertainty_w_u.py -m ./output/tandt_db_train_u/train --eval
#   python render_uncertainty_w_u.py -m ./output/tnt_sparse/Barn --split nvs_test \
#       --test_source /mnt/Data2/nerf_datasets/tnt_dpgs_gt_data/GT_data/Barn/nvs_test
#
# This file does not modify any existing module.
#

import os
import json
from argparse import ArgumentParser, Namespace
from os import makedirs

import struct

import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

from scene import Scene
from active.schema import override_test_idxs_dict
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.uncert_utils import compute_roc, compute_ause, plot_roc, write_auc
# VCURF in this module is the same formula as in render_uncertainty_w_Vcams_v2.py,
# but FisherRF there is optional, so we can skip it
from render_uncertainty_w_Vcams_and_fisherRF import render_uncertainty
from render_uncertainty_w_Vcams_v2 import plot_unmap
from train_w_uncertainty import GaussianModelWithUnc, render_confidence
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                 read_intrinsics_binary, read_intrinsics_text,
                                 read_next_bytes, qvec2rotmat)


# ---------------------------------------------------------------------------
# Sparse reference depth from the COLMAP model
# ---------------------------------------------------------------------------
def read_points3D_with_ids(sparse_dir):
    """dict point3D_id -> (xyz (3,), reprojection error). bin first, txt fallback."""
    pts = {}
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as fid:
            n = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(n):
                p = read_next_bytes(fid, 43, "QdddBBBd")
                track_len = read_next_bytes(fid, 8, "Q")[0]
                fid.seek(8 * track_len, 1)  # skip the track (image_id, point2D_idx) pairs
                pts[p[0]] = (np.array(p[1:4], dtype=np.float64), p[7])
    elif os.path.exists(txt_path):
        with open(txt_path) as fid:
            for line in fid:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e = line.split()
                pts[int(e[0])] = (np.array(list(map(float, e[1:4]))), float(e[7]))
    else:
        raise FileNotFoundError(f"no points3D.bin / points3D.txt in {sparse_dir}")
    return pts


class ColmapSparseDepth:
    """Projects the COLMAP sparse points seen by each image into that image."""

    def __init__(self, source_path, max_reproj_err=2.0):
        sparse_dir = os.path.join(source_path, "sparse/0")
        try:
            images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
            cams = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        except Exception:
            images = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
            cams = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
        pts = read_points3D_with_ids(sparse_dir)
        self.max_reproj_err = max_reproj_err
        self.points = pts
        self.cams = cams
        # same image name as scene/dataset_readers.py
        self.images = {os.path.basename(im.name).split(".")[0]: im for im in images.values()}
        print(f"[depth] COLMAP model: {len(self.images)} images, {len(pts)} points from {sparse_dir}")

    def __call__(self, view):
        """(H,W) float32 tensor of reference depth (0 = no point), or None."""
        im = self.images.get(view.image_name)
        if im is None:
            return None
        H, W = view.image_height, view.image_width
        cam = self.cams[im.camera_id]
        ids = im.point3D_ids
        keep = ids >= 0
        xys, ids = im.xys[keep], ids[keep]
        sel = [(i, k) for i, k in enumerate(ids) if k in self.points]
        if not sel:
            return None
        rows, ids = zip(*sel)
        xys = xys[list(rows)]
        X = np.stack([self.points[k][0] for k in ids])
        err = np.array([self.points[k][1] for k in ids])
        z = (X @ qvec2rotmat(im.qvec).T + im.tvec)[:, 2]
        u = np.floor(xys[:, 0] * W / cam.width).astype(np.int64)
        v = np.floor(xys[:, 1] * H / cam.height).astype(np.int64)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if self.max_reproj_err > 0:
            ok &= err <= self.max_reproj_err
        u, v, z = u[ok], v[ok], z[ok]
        depth = np.zeros((H, W), dtype=np.float32)
        order = np.argsort(-z)  # write far first, so the nearest point wins per pixel
        depth[v[order], u[order]] = z[order]
        return torch.from_numpy(depth).cuda()


def vcurf_key(n_vcam, r_scale):
    # key name produced by render_uncertainty() in render_uncertainty_w_Vcams_and_fisherRF.py
    return f'vcu({n_vcam} vcams, {r_scale} med)'


@torch.no_grad()
def render_ours(view, gaussians, pipeline):
    """Rendered uncertainty map of our method, u = 1 / C_hat (H,W)."""
    c_hat, acc, _ = render_confidence(view, gaussians, pipeline, for_vis=True)
    u = 1.0 / c_hat.clamp(min=gaussians.c_min, max=5.0)
    return u, acc


def load_nvs_test_views(test_source, images, split_file, split, resolution, data_device):
    """Cameras of a separate test set (same COLMAP frame as training), loaded like the training cameras."""
    sparse = os.path.join(test_source, "sparse/0")
    if os.path.isfile(os.path.join(sparse, "images.bin")):
        extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    else:
        extr = read_extrinsics_text(os.path.join(sparse, "images.txt"))
        intr = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    stem = lambda n: os.path.splitext(os.path.basename(str(n)))[0]
    image_dir = os.path.join(test_source, images)
    if split_file and os.path.isfile(split_file):
        wanted = {stem(n) for n in json.load(open(split_file))[split]}
        print(f"[nvs_test] {len(wanted)} '{split}' views from {split_file}")
    else:  # no split file: every image in the test image folder
        wanted = {stem(n) for n in os.listdir(image_dir)}
        print(f"[nvs_test] no split file, using the {len(wanted)} images in {image_dir}")
    extr = {k: e for k, e in extr.items() if stem(e.name) in wanted}
    missing = sorted(wanted - {stem(e.name) for e in extr.values()})
    if missing:
        print(f"[nvs_test] no camera in {sparse} for: {missing}")
    cam_infos = sorted(readColmapCameras(cam_extrinsics=extr, cam_intrinsics=intr, images_folder=image_dir),
                       key=lambda c: c.image_name)
    # same resizing as training (utils/camera_utils.py loadCam)
    return cameraList_from_camInfos(cam_infos, 1.0, Namespace(resolution=resolution, data_device=data_device))


class GtDepthMaps:
    """Dense GT depth <depth_dir>/<view>.npy (NaN / <= 0 = no surface), as a (H,W) tensor with 0 = invalid."""

    def __init__(self, depth_dir):
        self.depth_dir = depth_dir
        print(f"[depth] dense GT depth from {depth_dir}")

    def __call__(self, view):
        path = os.path.join(self.depth_dir, view.image_name + ".npy")
        if not os.path.isfile(path):
            return None
        d = np.load(path).astype(np.float32).squeeze()
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        d = torch.from_numpy(d).cuda()
        H, W = view.image_height, view.image_width
        if d.shape != (H, W):  # e.g. depth made at another --resolution: nearest resize to the render size
            print(f"[depth] {view.image_name}: GT depth {tuple(d.shape)} resized to {(H, W)}")
            d = torch.nn.functional.interpolate(d[None, None], size=(H, W), mode="nearest")[0, 0]
        return d


def evaluate(dataset, iteration, pipeline, args):
    gaussians = GaussianModelWithUnc(dataset.sh_degree)
    if args.override_idxs is not None:  # same special test split as render_uncertainty_w_Vcams_v2.py
        override_train_idxs = list(range(10_000))
        override_test_idxs = override_test_idxs_dict[args.override_idxs]
    else:  # same split as training (llffhold)
        override_train_idxs, override_test_idxs = None, None
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                  override_train_idxs=override_train_idxs, override_test_idxs=override_test_idxs)
    if args.split == "train":
        # quick check on a model trained without --eval: every train_every-th training view.
        # These views were seen in training, so the numbers are NOT a held-out evaluation.
        test_views = scene.getTrainCameras()[::args.train_every]
        print(f"[split=train] evaluating {len(test_views)} TRAINING views (not held out)")
    elif args.split == "nvs_test":
        if not args.test_source:
            raise RuntimeError("--split nvs_test needs --test_source (e.g. .../GT_data/<scene>/nvs_test)")
        split_file = args.test_split_file or os.path.join(args.test_source, "split_sparse_20.json")
        test_views = load_nvs_test_views(args.test_source, args.test_images, split_file, args.test_split,
                                         args.test_resolution, dataset.data_device)
        if not test_views:
            raise RuntimeError(f"no test views found in {args.test_source}")
        print(f"[nvs_test] evaluating {len(test_views)} views, "
              f"{test_views[0].image_width}x{test_views[0].image_height}")
    else:
        test_views = scene.getTestCameras()
        if not test_views:
            raise RuntimeError("No test cameras. Train and evaluate with --eval, or use --split train for a quick check.")
    if args.test_idxs is not None:
        test_views = [test_views[i] for i in args.test_idxs]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # as in v2
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    out_dir = os.path.join(dataset.model_path, args.save_dir, args.split)
    render_dir = os.path.join(out_dir, "renders")
    depth_dir = os.path.join(out_dir, "depth")
    error_dir = os.path.join(out_dir, "error")
    eval_dir = os.path.join(out_dir, f"eval_seed_{args.seed}")
    roc_dir = os.path.join(out_dir, f"roc_seed_{args.seed}")
    for d in (render_dir, depth_dir, error_dir, eval_dir, roc_dir):
        makedirs(d, exist_ok=True)

    # arguments for the baseline's render_uncertainty(): VCURF only, no FisherRF
    vc_args = Namespace(render_fisher=False, render_vcam=True,
                        n_vcam=[args.vcurf_n_vcam], r_scale=[args.vcurf_r_scale],
                        only_depth=False, only_rgb=False)
    vkey = vcurf_key(args.vcurf_n_vcam, args.vcurf_r_scale)
    names = {'rgb_err': 'rgb_err', vkey: vkey, 'ours_u': 'ours_u'}

    opt_label = 'rgb_err'
    ROCs, AUCs, AUSEs = {}, {}, {}
    # depth-error evaluation against COLMAP sparse depth
    depth_ref, depth_ref_name = None, None
    if not args.no_depth_eval:
        gt_depth_dir = args.gt_depth_dir or (os.path.join(args.test_source, "gt_depth")
                                             if args.split == "nvs_test" else None)
        if gt_depth_dir and os.path.isdir(gt_depth_dir):
            depth_ref, depth_ref_name = GtDepthMaps(gt_depth_dir), f"dense GT depth ({gt_depth_dir})"
        else:
            if gt_depth_dir:
                print(f"[depth] {gt_depth_dir} not found, falling back to COLMAP sparse depth")
            try:  # sparse reference: COLMAP points of the model the test views come from
                src = args.test_source if args.split == "nvs_test" else dataset.source_path
                depth_ref = ColmapSparseDepth(src, args.depth_max_reproj_err)
                depth_ref_name = f"COLMAP sparse points ({src})"
            except Exception as e:
                print(f"[depth] skip depth-error evaluation: {e}")
    d_opt_label = 'depth_err'
    d_names = {'depth_err': 'depth_err', vkey: vkey, 'ours_u': 'ours_u'}
    D_ROCs, D_AUCs, D_AUSEs = {}, {}, {}
    depth_skipped = []
    agree_ause = []  # AUSE of ours with VCURF as oracle
    per_view = []

    for idx, view in enumerate(tqdm(test_views, desc=f"Evaluating uncertainty on {args.split} views")):
        pred_img, _, depth, rests = render_uncertainty(view, gaussians, pipeline, background, None, vc_args)
        u_map, acc = render_ours(view, gaussians, pipeline)

        maps = {'rgb_err': rests['rgb_err'], vkey: rests[vkey], 'ours_u': u_map}
        mask = (depth > 0.).squeeze().detach()
        values = {k: maps[k].squeeze()[mask].flatten().float() for k in maps}

        rocs, aucs, auses = {}, {}, {}
        for k in values:
            label = names[k]
            roc, auc = compute_roc(opt=values[opt_label], est=values[k], intervals=20)
            _, ause = compute_ause(opt=values[opt_label], est=values[k], intervals=100)
            rocs[label], aucs[label], auses[label] = np.array(roc), auc, ause
            ROCs.setdefault(label, []).append(roc)
            AUCs.setdefault(label, []).append(auc)
            AUSEs.setdefault(label, []).append(ause)
        _, a = compute_ause(opt=values[vkey], est=values['ours_u'], intervals=100)
        agree_ause.append(a)

        # per-view ROC / AUC / AUSE, same file names as v2
        np.savez(os.path.join(roc_dir, '{0:05d}'.format(idx) + ".npz"), roc_dict=rocs)
        plot_roc(ROC_dict=dict(rocs), fig_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + ".jpg"),
                 opt_label=opt_label, intervals=20)
        write_auc(AUC_dict=aucs, txt_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + "auc.txt"))
        write_auc(AUC_dict=auses, txt_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + "ause.txt"))
        pv = {"idx": idx, "image": view.image_name,
              "auc": aucs, "ause": auses, "ours_vs_vcurf_ause": a}

        # ---- depth error vs COLMAP sparse depth ----
        gt_depth = depth_ref(view) if depth_ref is not None else None
        depth_err_map = None
        if gt_depth is not None:
            acc_s = acc.squeeze()
            pred_depth = depth.squeeze() / acc_s.clamp(min=1e-6)  # depth is sum(alpha*T*z)
            d_mask = mask & (gt_depth > 0) & (acc_s > args.depth_min_acc)
            n_pts = int(d_mask.sum())
            depth_err_map = torch.zeros_like(gt_depth)
            depth_err_map[d_mask] = (pred_depth[d_mask] - gt_depth[d_mask]).abs()
            if args.depth_err_type == "rel":
                depth_err_map[d_mask] = depth_err_map[d_mask] / gt_depth[d_mask]
            if n_pts >= args.depth_min_points:
                d_maps = {'depth_err': depth_err_map, vkey: maps[vkey], 'ours_u': u_map}
                d_values = {k: d_maps[k].squeeze()[d_mask].flatten().float() for k in d_maps}
                d_rocs, d_aucs, d_auses = {}, {}, {}
                for k in d_values:
                    label = d_names[k]
                    roc, auc = compute_roc(opt=d_values[d_opt_label], est=d_values[k], intervals=20)
                    _, ause = compute_ause(opt=d_values[d_opt_label], est=d_values[k], intervals=100)
                    d_rocs[label], d_aucs[label], d_auses[label] = np.array(roc), auc, ause
                    D_ROCs.setdefault(label, []).append(roc)
                    D_AUCs.setdefault(label, []).append(auc)
                    D_AUSEs.setdefault(label, []).append(ause)
                plot_roc(ROC_dict=dict(d_rocs), fig_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + "_depth.jpg"),
                         opt_label=d_opt_label, intervals=20)
                write_auc(AUC_dict=d_aucs, txt_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + "depth_auc.txt"))
                write_auc(AUC_dict=d_auses, txt_name=os.path.join(roc_dir, '{0:05d}'.format(idx) + "depth_ause.txt"))
                pv.update({"depth_auc": d_aucs, "depth_ause": d_auses, "depth_n_points": n_pts,
                           "depth_mae": float(d_values['depth_err'].mean())})
            else:
                depth_skipped.append(view.image_name)
                pv["depth_n_points"] = n_pts
        else:
            depth_skipped.append(view.image_name)
        per_view.append(pv)

        # images, same naming / colormap / quantile clipping as v2
        mask_cpu = mask.cpu()
        name = view.image_name
        torchvision.utils.save_image(pred_img.detach().clamp(0, 1), os.path.join(render_dir, f"{args.split}_{name}.png"))
        plot_unmap(depth.squeeze().detach().cpu(), mask_cpu, os.path.join(depth_dir, f"{name}.jpg"), cmap='magma', q=0.95)
        plot_unmap(maps['rgb_err'].squeeze().cpu(), mask_cpu, os.path.join(error_dir, f"{name}.jpg"), q=0.8)
        if depth_err_map is not None and bool(((gt_depth > 0).cpu() & mask_cpu).any()):
            plot_unmap(depth_err_map.cpu(), (gt_depth > 0).cpu() & mask_cpu,
                       os.path.join(error_dir, f"{name}_depth_err.jpg"), q=0.95)
        plot_unmap(maps[vkey].squeeze().cpu(), mask_cpu,
                   os.path.join(eval_dir, f"vcurf_{args.vcurf_n_vcam}_vcams_{args.vcurf_r_scale}_med_{name}.jpg"), q=0.8)
        plot_unmap(u_map.cpu(), mask_cpu, os.path.join(eval_dir, f"ours_u_{name}.jpg"), q=0.8)
        plot_unmap(torch.log(u_map).cpu(), mask_cpu, os.path.join(eval_dir, f"ours_logu_{name}.jpg"), q=0.95)

        # raw maps
        np.savez(os.path.join(eval_dir, f"uncertainty_{idx:03d}_{name}.npz"),
                 rgb_err=maps['rgb_err'].squeeze().cpu().numpy(),
                 vcurf=maps[vkey].squeeze().cpu().numpy(),
                 ours_u=u_map.cpu().numpy(),
                 acc=acc.cpu().numpy(),
                 depth=depth.squeeze().cpu().numpy(),
                 mask=mask_cpu.numpy(),
                 ref_depth=(gt_depth.cpu().numpy() if gt_depth is not None else np.zeros(0)),
                 depth_err=(depth_err_map.cpu().numpy() if depth_err_map is not None else np.zeros(0)))

    # summary over test views, same file names as v2 (test.png / test.txt / testause.txt)
    mean_roc = {k: np.array(v).mean(0) for k, v in ROCs.items()}
    mean_auc = {k: float(np.mean(v)) for k, v in AUCs.items()}
    mean_ause = {k: float(np.mean(v)) for k, v in AUSEs.items()}
    plot_roc(ROC_dict=dict(mean_roc), fig_name=os.path.join(roc_dir, f"{args.split}.png"), opt_label=opt_label, intervals=20)
    write_auc(AUC_dict=mean_auc, txt_name=os.path.join(roc_dir, f"{args.split}.txt"))
    write_auc(AUC_dict=mean_ause, txt_name=os.path.join(roc_dir, f"{args.split}ause.txt"))
    summary = {"model_path": dataset.model_path, "iteration": scene.loaded_iter, "split": args.split, "n_views": len(test_views),
               "vcurf": {"n_vcam": args.vcurf_n_vcam, "r_scale": args.vcurf_r_scale},
               "oracle": opt_label, "mean_auc": mean_auc, "mean_ause": mean_ause,
               "ours_vs_vcurf_ause_mean": float(np.mean(agree_ause)), "per_view": per_view}
    if D_AUCs:
        d_mean_roc = {k: np.array(v).mean(0) for k, v in D_ROCs.items()}
        d_mean_auc = {k: float(np.mean(v)) for k, v in D_AUCs.items()}
        d_mean_ause = {k: float(np.mean(v)) for k, v in D_AUSEs.items()}
        plot_roc(ROC_dict=dict(d_mean_roc), fig_name=os.path.join(roc_dir, f"{args.split}_depth.png"),
                 opt_label=d_opt_label, intervals=20)
        write_auc(AUC_dict=d_mean_auc, txt_name=os.path.join(roc_dir, f"{args.split}_depth.txt"))
        write_auc(AUC_dict=d_mean_ause, txt_name=os.path.join(roc_dir, f"{args.split}_depthause.txt"))
        summary.update({"depth_oracle": f"{args.depth_err_type} depth error vs {depth_ref_name}", "depth_n_views": len(D_AUCs[d_opt_label]),
                        "depth_skipped_views": depth_skipped, "depth_mean_auc": d_mean_auc, "depth_mean_ause": d_mean_ause})
    with open(os.path.join(roc_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.split} views: {len(test_views)}   (oracle: {opt_label}, lower AUC/AUSE is better)")
    print(f"{'method':<24}{'AUC':>12}{'AUSE':>12}")
    for k in mean_auc:
        print(f"{k:<24}{mean_auc[k]:>12.6f}{mean_ause[k]:>12.6f}")
    print(f"AUSE of ours w.r.t. VCURF: {np.mean(agree_ause):.6f}")
    if D_AUCs:
        print(f"\n(oracle: {args.depth_err_type} depth error vs {depth_ref_name}, {len(D_AUCs[d_opt_label])} views"
              f"{', skipped ' + str(len(depth_skipped)) if depth_skipped else ''})")
        print(f"{'method':<24}{'AUC':>12}{'AUSE':>12}")
        for k in d_mean_auc:
            print(f"{k:<24}{d_mean_auc[k]:>12.6f}{d_mean_ause[k]:>12.6f}")
    else:
        print("\n[depth] no depth-error results (no COLMAP model or too few points)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate per-Gaussian uncertainty against VCURF")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_idxs", nargs="+", default=None, type=int, help="subset of test views")
    parser.add_argument("--override_idxs", default=None, type=str,
                        help="special test split from active.schema.override_test_idxs_dict (as in v2); default = training split")
    parser.add_argument("--save_dir", default="eval_u", type=str)
    parser.add_argument("--split", default="test", choices=["test", "train", "nvs_test"],
                        help="test = llffhold views (needs --eval); nvs_test = separate test set (--test_source); "
                             "train = quick check on training views")
    parser.add_argument("--test_source", default=None, type=str,
                        help="nvs_test folder with sparse/0, images/, split json and gt_depth/")
    parser.add_argument("--test_images", default="images", type=str, help="image folder inside --test_source")
    parser.add_argument("--test_split_file", default=None, type=str,
                        help="default: <test_source>/split_sparse_20.json (all images if it does not exist)")
    parser.add_argument("--test_split", default="test", type=str, help="which list of the split file")
    parser.add_argument("--test_resolution", default=2, type=int,
                        help="resolution of the nvs_test views (like -r); 2 matches gt_depth from "
                             "gt_depthmaps_standalone.py -r 2")
    parser.add_argument("--gt_depth_dir", default=None, type=str,
                        help="dense GT depth <view>.npy; default for nvs_test: <test_source>/gt_depth")
    parser.add_argument("--depth_err_type", default="abs", choices=["abs", "rel"],
                        help="depth error oracle: |d - d_gt| or |d - d_gt| / d_gt")
    parser.add_argument("--train_every", default=8, type=int, help="with --split train, use every N-th training view")
    # names differ from the training args so cfg_args cannot override them
    parser.add_argument("--vcurf_n_vcam", default=6, type=int, help="virtual cameras for the VCURF baseline")
    parser.add_argument("--vcurf_r_scale", default=0.1, type=float, help="sphere radius / median depth for VCURF")
    parser.add_argument("--no_depth_eval", action="store_true", help="skip the depth-error evaluation")
    parser.add_argument("--depth_max_reproj_err", default=2.0, type=float,
                        help="drop COLMAP points with larger reprojection error (px); <=0 keeps all")
    parser.add_argument("--depth_min_points", default=50, type=int,
                        help="skip depth AUC/AUSE of a view with fewer valid sparse points")
    parser.add_argument("--depth_min_acc", default=0.5, type=float,
                        help="only pixels with rendered opacity above this are used for depth error")
    args = get_combined_args(parser)
    # get_combined_args drops command-line args whose value is None, so options
    # left at a None default are missing from args; restore them
    for k in ("override_idxs", "test_idxs", "test_source", "test_split_file", "gt_depth_dir"):
        if not hasattr(args, k):
            setattr(args, k, None)
    print("Evaluating " + args.model_path)

    safe_state(args.quiet, seed=args.seed)
    evaluate(model.extract(args), args.iteration, pipeline.extract(args), args)
