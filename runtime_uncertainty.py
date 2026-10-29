#
# Inference runtime of the uncertainty methods on the test views of one trained scene:
#   render    plain 3DGS RGB(+depth) render, for reference
#   VCURF     render + N virtual cameras + backward warping (render_uncertainty_w_Vcams_and_fisherRF.py)
#   FisherRF  one-time pass over the training views to accumulate the per-Gaussian Hessian
#             (one backward per view), then per test view: render + one render with the Hessian as color
#   WarpRF    render + render every training view + backward warping into the test view
#             (render_uncertainty_warprf.py; training views from --training_views yaml if given)
#   Ensemble  render each of M independently trained models (render_uncertainty_w_emsemble.py)
#
# Only inference is timed (model loading, metric computation and file I/O are not), with
# torch.cuda.synchronize() around every call. Every method also produces the RGB image.
# Nothing is written except the result json / txt (--out).
#
# Example:
#   python runtime_uncertainty.py -m <model> --n_vcam 2 4 6 8 \
#       --training_views mip360_training_views.yaml --scene garden \
#       --ensemble_paths <model_seed0> <model_seed500> ...
#
import os
import sys
import json
import time
import itertools
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from einops import reduce, repeat

from scene import Scene
from gaussian_renderer import GaussianModel, modified_render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import BackwardWarping, extract_scene_center_and_C2W
from render_uncertainty_w_Vcams_and_fisherRF import render_uncertainty, capture


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


# ---------------------------------------------------------------- methods (one test view each)
@torch.no_grad()
def plain_render(view, g, pipe, bg):
    pkg = modified_render(view, g, pipe, bg)
    return pkg["render"], pkg["depth"]


@torch.no_grad()
def vcurf(view, g, pipe, bg, n_vcam, r_scale):
    a = Namespace(render_fisher=False, render_vcam=True, n_vcam=[n_vcam], r_scale=[r_scale],
                  only_depth=False, only_rgb=False)
    return render_uncertainty(view, g, pipe, bg, None, a)


def fisher_precompute(views, g, pipe, bg):
    """Per-Gaussian Hessian diagonal over the given views, as in render_uncertainty_w_Vcams_and_fisherRF.py."""
    params = capture(g)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    drop = [name2idx[k] for k in ["rotation", "scale", "xyz", "opacity"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in drop]
    optim = torch.optim.SGD(params, 0.)
    g.optimizer = optim
    H = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)
    for view in views:
        pred = modified_render(view, g, pipe, bg)["render"]
        pred.backward(gradient=torch.ones_like(pred), retain_graph=True)
        H += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
        optim.zero_grad(set_to_none=True)
    for p in params:
        p.requires_grad_(False)
    return repeat(H.detach(), "n -> n c", c=3)


@torch.no_grad()
def fisher_view(view, g, pipe, bg, hessian_color):
    xyz = g.get_xyz
    pts = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1) @ view.world_view_transform
    cur = hessian_color * pts[:, 2, None].clamp(min=0)
    a = Namespace(render_fisher=True, render_vcam=False)
    return render_uncertainty(view, g, pipe, bg, cur, a)


@torch.no_grad()
def warprf(view, src_views, g, pipe, bg):
    """render_uncertainty_warprf.py without its extra FisherRF pass."""
    pkg = modified_render(view, g, pipe, bg)
    pred_img, depth = pkg["render"], pkg["depth"]
    _, rd_c2w = extract_scene_center_and_C2W(depth, view)
    rd_c2w = rd_c2w.to(depth.device)
    K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                           fovX=view.FoVx, fovY=view.FoVy).to(depth.device)
    backwarp = BackwardWarping(out_hw=(view.image_height, view.image_width), device=depth.device, K=K)
    N = len(src_views)
    rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
    rd_depths = rd_depth.repeat(N, 1, 1, 1)
    depths, imgs, T = [], [], []
    for sv in src_views:
        p = modified_render(sv, g, pipe, bg)
        T.append(sv.world_view_transform.transpose(0, 1) @ rd_c2w)
        depths.append(p["depth"].unsqueeze(0))
        imgs.append(p["render"])
    w_imgs, w_depths, nv_mask = backwarp(img_src=torch.stack(imgs), depth_src=torch.stack(depths),
                                         depth_tgt=rd_depths, tgt2src_transform=torch.stack(T))
    numels = float(N) - nv_mask.sum(0)
    wd = torch.zeros_like(rd_depth.squeeze(0))
    wd[numels > 0] = w_depths.sum(0)[numels > 0] / numels[numels > 0]
    d_l2 = ((rd_depth.squeeze(0) - wd) ** 2).squeeze(0)
    d_l2 = (d_l2 - d_l2.min()) / (d_l2.max() - d_l2.min())
    rend = pred_img.mean(0, keepdim=True)
    wp = torch.zeros_like(rend)
    wp[numels > 0] = w_imgs.sum(0).mean(0, keepdim=True)[numels > 0] / numels[numels > 0]
    c_l2 = ((rend - wp) ** 2).squeeze(0)
    c_l2 = (c_l2 - c_l2.min()) / (c_l2.max() - c_l2.min())
    return pred_img, d_l2 + c_l2


@torch.no_grad()
def ensemble(view, models, pipe, bg):
    imgs, depths = [], []
    for m in models:
        p = modified_render(view, m, pipe, bg)
        imgs.append(p["render"])
        depths.append(p["depth"])
    imgs, depths = torch.stack(imgs), torch.stack(depths)
    return imgs.mean(0), torch.std(imgs, 0).mean(0), torch.std(depths, 0)


# ---------------------------------------------------------------- timing
def time_method(name, fn, views, warmup):
    for _ in range(warmup):
        fn(views[0])
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for v in views:
        _, t = sync_time(lambda: fn(v))
        ts.append(t)
    ts = np.array(ts)
    r = {"sec_per_view_mean": float(ts.mean()), "sec_per_view_std": float(ts.std()),
         "fps": float(1.0 / ts.mean()), "n_views": len(ts),
         "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    print(f"{name:<28} {ts.mean():9.4f} s/view  (std {ts.std():.4f})  {1 / ts.mean():8.2f} FPS  "
          f"peak {r['peak_mem_GB']:.2f} GB")
    return r


def load_model(path, iteration, sh_degree):
    if iteration == -1:
        its = [int(d.split("_")[-1]) for d in os.listdir(os.path.join(path, "point_cloud"))]
        iteration = max(its)
    m = GaussianModel(sh_degree)
    m.load_ply(os.path.join(path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    return m


def main(dataset, pipe, args):
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    if not test_views:
        test_views = train_views[::8]
        print("[warn] no test cameras (model not trained with --eval): timing every 8th training view")
    if args.max_views:
        test_views = test_views[:args.max_views]
    v0 = test_views[0]
    info = {"model_path": dataset.model_path, "iteration": scene.loaded_iter, "gpu": torch.cuda.get_device_name(0),
            "n_gaussians": int(g.get_xyz.shape[0]), "resolution": [v0.image_width, v0.image_height],
            "n_test_views": len(test_views), "n_train_views": len(train_views)}
    print(json.dumps(info, indent=2))
    res = {}

    res["render (3DGS)"] = time_method("render (3DGS)", lambda v: plain_render(v, g, pipe, bg), test_views, args.warmup)

    for n in args.n_vcam:
        k = f"VCURF ({n} vcams)"
        res[k] = time_method(k, lambda v: vcurf(v, g, pipe, bg, n, args.r_scale), test_views, args.warmup)

    # FisherRF: one-time Hessian over the training views + per-view render
    fisher_views = list(train_views) + (list(test_views) if args.fisher_include_test else [])
    fisher_precompute(fisher_views[:1], g, pipe, bg)  # warm-up
    H, t_pre = sync_time(lambda: fisher_precompute(fisher_views, g, pipe, bg))
    k = "FisherRF (per view)"
    r = time_method(k, lambda v: fisher_view(v, g, pipe, bg, H), test_views, args.warmup)
    r.update({"precompute_sec": t_pre, "precompute_n_views": len(fisher_views),
              "amortized_sec_per_view": r["sec_per_view_mean"] + t_pre / len(test_views),
              "first_view_sec": r["sec_per_view_mean"] + t_pre})
    print(f"{'FisherRF Hessian (once)':<28} {t_pre:9.4f} s over {len(fisher_views)} views -> amortized "
          f"{r['amortized_sec_per_view']:.4f} s/view ({1 / r['amortized_sec_per_view']:.2f} FPS) over "
          f"{len(test_views)} test views; {r['first_view_sec']:.4f} s for a single new view")
    res["FisherRF"] = r
    del H

    # WarpRF: source views = training views
    src = list(train_views)
    if args.training_views and args.scene:
        import yaml
        names = yaml.full_load(open(args.training_views))[args.scene]
        sel = [c for c in train_views if c.image_name in names]
        if sel:
            src = sel
    k = f"WarpRF ({len(src)} src views)"
    res["WarpRF"] = time_method(k, lambda v: warprf(v, src, g, pipe, bg), test_views, args.warmup)
    res["WarpRF"]["n_src_views"] = len(src)

    # Ensemble
    if args.ensemble_paths:
        models = [load_model(p, args.ensemble_iteration, dataset.sh_degree) for p in args.ensemble_paths]
        ens_note = "independently trained members: " + ", ".join(args.ensemble_paths)
    else:
        models = [g] * args.n_ensemble
        ens_note = f"no members given: the same model rendered {args.n_ensemble} times (same #Gaussians)"
    k = f"Ensemble ({len(models)} models)"
    res["Ensemble"] = time_method(k, lambda v: ensemble(v, models, pipe, bg), test_views, args.warmup)
    res["Ensemble"].update({"n_models": len(models), "note": ens_note,
                            "member_n_gaussians": [int(m.get_xyz.shape[0]) for m in models]})

    out = args.out or os.path.join(dataset.model_path, "runtime_uncertainty.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"info": info, "results": res, "args": {k: v for k, v in vars(args).items()
                                                            if isinstance(v, (int, float, str, list, type(None)))}},
                  f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Runtime of uncertainty methods")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_vcam", nargs="+", default=[6], type=int)
    parser.add_argument("--r_scale", default=0.1, type=float)
    parser.add_argument("--training_views", default=None, type=str, help="yaml {scene: [image names]} of the WarpRF source views")
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--ensemble_paths", nargs="+", default=None, type=str, help="model folders of the ensemble members")
    parser.add_argument("--ensemble_iteration", default=-1, type=int)
    parser.add_argument("--n_ensemble", default=5, type=int, help="members to simulate when --ensemble_paths is not given")
    parser.add_argument("--fisher_include_test", action="store_true",
                        help="also accumulate the Hessian over the test views (the original script does)")
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--out", default=None, type=str)
    args = get_combined_args(parser)
    for k, v in (("training_views", None), ("scene", None), ("ensemble_paths", None), ("out", None)):
        if not hasattr(args, k):
            setattr(args, k, v)
    safe_state(args.quiet, seed=args.seed)
    main(model.extract(args), pipeline.extract(args), args)
