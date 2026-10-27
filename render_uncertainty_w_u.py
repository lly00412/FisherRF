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
# Output layout follows render_uncertainty_w_Vcams_v2.py, under <model_path>/<save_dir>/:
#   renders/  depth/  error/  eval_seed_<s>/  roc_seed_<s>/
#
# The model must be trained with --eval, otherwise there is no test set.
#
# Example:
#   python render_uncertainty_w_u.py -m ./output/tandt_db_train_u/train --eval
#
# This file does not modify any existing module.
#

import os
import json
from argparse import ArgumentParser, Namespace
from os import makedirs

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


def vcurf_key(n_vcam, r_scale):
    # key name produced by render_uncertainty() in render_uncertainty_w_Vcams_and_fisherRF.py
    return f'vcu({n_vcam} vcams, {r_scale} med)'


@torch.no_grad()
def render_ours(view, gaussians, pipeline):
    """Rendered uncertainty map of our method, u = 1 / C_hat (H,W)."""
    c_hat, acc, _ = render_confidence(view, gaussians, pipeline, for_vis=True)
    u = 1.0 / c_hat.clamp(min=gaussians.c_min, max=5.0)
    return u, acc


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
        per_view.append({"idx": idx, "image": view.image_name,
                         "auc": aucs, "ause": auses, "ours_vs_vcurf_ause": a})

        # images, same naming / colormap / quantile clipping as v2
        mask_cpu = mask.cpu()
        name = view.image_name
        torchvision.utils.save_image(pred_img.detach().clamp(0, 1), os.path.join(render_dir, f"{args.split}_{name}.png"))
        plot_unmap(depth.squeeze().detach().cpu(), mask_cpu, os.path.join(depth_dir, f"{name}.jpg"), cmap='magma', q=0.95)
        plot_unmap(maps['rgb_err'].squeeze().cpu(), mask_cpu, os.path.join(error_dir, f"{name}.jpg"), q=0.8)
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
                 mask=mask_cpu.numpy())

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
    with open(os.path.join(roc_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.split} views: {len(test_views)}   (oracle: {opt_label}, lower AUC/AUSE is better)")
    print(f"{'method':<24}{'AUC':>12}{'AUSE':>12}")
    for k in mean_auc:
        print(f"{k:<24}{mean_auc[k]:>12.6f}{mean_ause[k]:>12.6f}")
    print(f"AUSE of ours w.r.t. VCURF: {np.mean(agree_ause):.6f}")
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
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="test = held-out views (needs --eval); train = quick check on training views")
    parser.add_argument("--train_every", default=8, type=int, help="with --split train, use every N-th training view")
    # names differ from the training args so cfg_args cannot override them
    parser.add_argument("--vcurf_n_vcam", default=6, type=int, help="virtual cameras for the VCURF baseline")
    parser.add_argument("--vcurf_r_scale", default=0.1, type=float, help="sphere radius / median depth for VCURF")
    args = get_combined_args(parser)
    # get_combined_args drops command-line args whose value is None, so options
    # left at a None default are missing from args; restore them
    for k in ("override_idxs", "test_idxs"):
        if not hasattr(args, k):
            setattr(args, k, None)
    print("Evaluating " + args.model_path)

    safe_state(args.quiet, seed=args.seed)
    evaluate(model.extract(args), args.iteration, pipeline.extract(args), args)
