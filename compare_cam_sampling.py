#
# Training-camera vs virtual-camera sampling under the same discrepancy, normalization and
# rendering budget (rebuttal experiment).
#
# For every test view the reference RGB + depth are rendered once. Then K source cameras are
# rendered, warped into the test view (utils/proj_utils.BackwardWarping) and the same VCURF score
# is computed for every source set:
#     u = minmax(depth_l2) + minmax(rgb_l2),  l2 against the mean over the sources that see the pixel
# (identical to render_uncertainty_w_Vcams_and_fisherRF.py and render_uncertainty_warprf.py).
# Only the source cameras differ:
#     vcam      K virtual cameras on a sphere of radius r_scale * median depth around the test camera,
#               looking at the scene centre (VCURF)
#     train_nn  the K training cameras closest to the test camera (camera centre distance)
#     train_rnd K random training cameras
#     warprf    all training cameras (WarpRF as in the paper, budget = #training views)
# Training cameras = the views the model was trained on (--training_views yaml), else all train cameras.
# Metrics: AUC / AUSE against the per-pixel RGB MSE on foreground pixels (depth > 0), as in the
# other evaluation scripts, plus coverage = fraction of foreground pixels seen by >= 1 source.
# Nothing is written except --out.
#
import os
import json
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from scene import Scene
from scene.cameras import VirtualCam
from gaussian_renderer import GaussianModel, modified_render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import BackwardWarping, extract_scene_center_and_C2W
from utils.uncert_utils import compute_roc, compute_ause


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


@torch.no_grad()
def discrepancy(view, pred_img, depth, rd_c2w, src_views, g, pipe, bg):
    """VCURF score of the test view from the given source cameras. Returns (u, coverage mask)."""
    K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                           fovX=view.FoVx, fovY=view.FoVy).to(depth.device)
    backwarp = BackwardWarping(out_hw=(view.image_height, view.image_width), device=depth.device, K=K)
    N = len(src_views)
    rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
    depths, imgs, T = [], [], []
    for sv in src_views:
        p = modified_render(sv, g, pipe, bg)
        T.append(sv.world_view_transform.transpose(0, 1) @ rd_c2w)
        depths.append(p["depth"].unsqueeze(0))
        imgs.append(p["render"])
    w_imgs, w_depths, nv_mask = backwarp(img_src=torch.stack(imgs), depth_src=torch.stack(depths),
                                         depth_tgt=rd_depth.repeat(N, 1, 1, 1), tgt2src_transform=torch.stack(T))
    numels = float(N) - nv_mask.sum(0)
    wd = torch.zeros_like(rd_depth.squeeze(0))
    wd[numels > 0] = w_depths.sum(0)[numels > 0] / numels[numels > 0]
    d_l2 = ((rd_depth.squeeze(0) - wd) ** 2).squeeze(0)
    rend = pred_img.mean(0, keepdim=True)
    wp = torch.zeros_like(rend)
    wp[numels > 0] = w_imgs.sum(0).mean(0, keepdim=True)[numels > 0] / numels[numels > 0]
    c_l2 = ((rend - wp) ** 2).squeeze(0)
    return minmax(d_l2) + minmax(c_l2), (numels > 0).squeeze()


def main(dataset, pipe, args):
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    if args.max_views:
        test_views = test_views[:args.max_views]
    pool = list(train_views)
    if args.training_views and args.scene:
        import yaml
        names = yaml.full_load(open(args.training_views))[args.scene]
        sel = [c for c in train_views if c.image_name in names]
        if sel:
            pool = sel
    centers = torch.stack([c.camera_center for c in pool])
    print(f"{len(test_views)} test views, {len(pool)} training cameras as the pool, K = {args.ks}")

    rng = np.random.default_rng(args.seed)
    acc = {}  # method -> list of (auc, ause, coverage, time)

    def add(name, u, cov, mask, rgb_err, t):
        est = u[mask].flatten().float()
        opt = rgb_err[mask].flatten().float()
        _, auc = compute_roc(opt=opt, est=est, intervals=20)
        _, ause = compute_ause(opt=opt, est=est, intervals=100)
        acc.setdefault(name, []).append((auc, ause, float(cov[mask].float().mean()), t))

    for vi, view in enumerate(test_views):
        with torch.no_grad():
            pkg = modified_render(view, g, pipe, bg)
            pred_img, depth = pkg["render"], pkg["depth"]
            rgb_err = torch.mean((pred_img - view.original_image[0:3].cuda()) ** 2, 0)
            mask = (depth > 0).squeeze()
            look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
            rd_c2w = rd_c2w.to(depth.device)
            d_med = depth.flatten().median()
            # oracle row for reference
            add("oracle(rgb_err)", rgb_err, torch.ones_like(mask), mask, rgb_err, 0.0)
            dist = torch.linalg.norm(centers - view.camera_center[None], dim=1)
            order = torch.argsort(dist).tolist()

            def run(name, srcs):
                torch.cuda.synchronize(); t = time.perf_counter()
                u, cov = discrepancy(view, pred_img, depth, rd_c2w, srcs, g, pipe, bg)
                torch.cuda.synchronize()
                add(name, u, cov, mask, rgb_err, time.perf_counter() - t)

            for k in args.ks:
                for s in range(args.n_seeds):
                    vc = VirtualCam(view).get_N_near_cam_by_look_at(k, look_at=look_at, radiaus=args.r_scale * d_med)
                    run(f"vcam K={k}", vc)
                run(f"train_nn K={k}", [pool[i] for i in order[:k]])
                for s in range(args.n_seeds):
                    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
                    run(f"train_rnd K={k}", [pool[i] for i in idx])
            run(f"warprf (all {len(pool)} train)", pool)
        print(f"view {vi + 1}/{len(test_views)} done")

    res = {}
    print(f"\n{'method':<26}{'AUC':>10}{'AUSE':>10}{'coverage':>10}{'s/view':>10}   (lower AUC/AUSE is better)")
    for name, rows in acc.items():
        a = np.array(rows)
        res[name] = {"auc": float(a[:, 0].mean()), "ause": float(a[:, 1].mean()),
                     "coverage": float(a[:, 2].mean()), "sec": float(a[:, 3].mean()), "n": len(rows)}
        r = res[name]
        print(f"{name:<26}{r['auc']:>10.4f}{r['ause']:>10.4f}{r['coverage']:>10.3f}{r['sec']:>10.4f}")
    info = {"model_path": dataset.model_path, "iteration": scene.loaded_iter, "n_test": len(test_views),
            "pool": [c.image_name for c in pool], "ks": args.ks, "r_scale": args.r_scale, "n_seeds": args.n_seeds}
    out = args.out or os.path.join(dataset.model_path, "compare_cam_sampling.json")
    with open(out, "w") as f:
        json.dump({"info": info, "results": res}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="training vs virtual camera sampling")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ks", nargs="+", default=[2, 4, 6, 8], type=int, help="rendering budgets (source cameras)")
    parser.add_argument("--r_scale", default=0.1, type=float)
    parser.add_argument("--n_seeds", default=3, type=int, help="repeats of the random samplings (vcam, train_rnd)")
    parser.add_argument("--training_views", default=None, type=str)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = get_combined_args(parser)
    for k in ("training_views", "scene", "out"):
        if not hasattr(args, k):
            setattr(args, k, None)
    safe_state(args.quiet, seed=args.seed)
    main(model.extract(args), pipeline.extract(args), args)
