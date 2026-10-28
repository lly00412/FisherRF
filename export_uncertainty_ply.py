#
# Save a copy of a trained Gaussian point cloud whose colors show the per-Gaussian
# uncertainty u = exp(-unc) (from train_w_uncertainty.py), colored with matplotlib's
# "plasma" colormap (purple = low u, pink, yellow = high u).
#
# The output is a standard 3DGS ply (same geometry / opacity / scale / rotation,
# SH DC = colormap color, higher SH bands = 0), so any 3DGS viewer shows it.
# Gaussians that were never supervised (unc_observed = 0) get the top color.
# log(u) of the observed Gaussians is clipped to its [lo, hi] percentiles before mapping.
#
# train_w_uncertainty.py calls this whenever it saves a point cloud. For a ply that
# already exists:
#   python export_uncertainty_ply.py <model_path>/point_cloud/iteration_30000/point_cloud.ply
# writes uncertainty_point_cloud.ply (+ the colorbar) next to it.
#
# Only numpy / plyfile / matplotlib, no GPU needed.
#

import os
import json
from argparse import ArgumentParser

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814
EXTRA = ("unc", "unc_observed")  # our fields, left out so standard viewers read the file


def export_uncertainty_ply(ply_path, out_path=None, cmap="plasma", lo=1.0, hi=99.0, save_colorbar=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = os.path.join(os.path.dirname(ply_path), "uncertainty_point_cloud.ply")
    v = PlyData.read(ply_path).elements[0]
    names = [p.name for p in v.properties]
    if "unc" not in names:
        raise ValueError(f"{ply_path} has no 'unc' field (not trained with train_w_uncertainty.py)")

    log_u = -np.asarray(v["unc"], dtype=np.float64)  # u = exp(-unc)
    observed = (np.asarray(v["unc_observed"]) > 0.5) if "unc_observed" in names else np.ones_like(log_u, bool)
    ref = log_u[observed] if observed.any() else log_u
    vmin, vmax = np.percentile(ref, lo), np.percentile(ref, hi)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    t = np.clip((log_u - vmin) / (vmax - vmin), 0.0, 1.0)
    t[~observed] = 1.0
    rgb = matplotlib.colormaps[cmap](t)[:, :3]
    f_dc = (rgb - 0.5) / SH_C0  # inverse of RGB2SH's DC term

    keep = [n for n in names if n not in EXTRA]
    out = np.empty(len(log_u), dtype=[(n, "f4") for n in keep])
    for n in keep:
        if n in ("f_dc_0", "f_dc_1", "f_dc_2"):
            out[n] = f_dc[:, int(n[-1])]
        elif n.startswith("f_rest_"):
            out[n] = 0.0  # no view dependence
        else:
            out[n] = v[n]
    PlyData([PlyElement.describe(out, "vertex")]).write(out_path)

    info = {"source": os.path.abspath(ply_path), "cmap": cmap, "value": "log(u), u = exp(-unc)",
            "log_u_min": float(vmin), "log_u_max": float(vmax), "u_min": float(np.exp(vmin)), "u_max": float(np.exp(vmax)),
            "percentiles": [lo, hi], "n_points": int(len(log_u)), "n_unobserved": int((~observed).sum()),
            "unobserved_color": "top of colormap"}
    stem = os.path.splitext(out_path)[0]
    with open(stem + "_colorbar.json", "w") as f:
        json.dump(info, f, indent=2)
    if save_colorbar:
        fig, ax = plt.subplots(figsize=(1.2, 4))
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap),
                     cax=ax, label="log u")
        fig.savefig(stem + "_colorbar.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    print(f"Saved uncertainty point cloud to {out_path} (log u in [{vmin:.3f}, {vmax:.3f}], "
          f"{info['n_unobserved']} unobserved)")
    return out_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Color a trained Gaussian point cloud by its uncertainty")
    parser.add_argument("ply", help="point_cloud.ply saved by train_w_uncertainty.py")
    parser.add_argument("--out", default=None, help="default: uncertainty_point_cloud.ply next to the input")
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--lo", default=1.0, type=float, help="lower percentile of log u for the colormap")
    parser.add_argument("--hi", default=99.0, type=float, help="upper percentile of log u for the colormap")
    a = parser.parse_args()
    export_uncertainty_ply(a.ply, a.out, a.cmap, a.lo, a.hi)
