'''
refer to https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/image_utils.py
'''
import torch
import matplotlib.pyplot as plt
def colormap(map, cmap="turbo",max=None, min=None):
    if max==None:
        max = map.max()
    if min==None:
        min = map.min()
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map[map>max] = max
    map[map<min] = min
    map = (map - min) / (max - min)
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration: the columns you care about ---
COLUMNS = [
    "d_mean", "d_var", "d_skewness", "d_kurtosis",
    "c_mean", "c_var", "c_skewness", "c_kurtosis", "psnr"
]

def make_corr_heatmap(
    csv_path: str,
    out_png: str,
    save_corr_csv: bool = False,
    corr_csv_path: str | None = None,
):
    """
    Load a CSV, compute Pearson correlation across the specified columns,
    save an annotated heatmap (matplotlib only), and optionally save the
    correlation matrix as a CSV.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    out_png : str
        Output filename for the heatmap image (e.g., 'bicycle.png').
    save_corr_csv : bool
        If True, also saves correlation matrix as CSV.
    corr_csv_path : str | None
        Path to save corr CSV. If None and save_corr_csv=True, uses out_png with '.csv'.
    """
    # --- Load data ---
    df = pd.read_csv(csv_path)

    # --- Ensure required columns exist ---
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # --- Coerce to numeric (in case of strings), then drop rows with NA on these columns ---
    df_sel = df[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()

    # --- Compute correlation matrix ---
    corr = df_sel.corr(method="pearson")

    # --- Plot heatmap (annotated) with matplotlib only ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient")

    # Ticks and labels
    ax.set_xticks(np.arange(len(COLUMNS)))
    ax.set_yticks(np.arange(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS, rotation=45, ha="right")
    ax.set_yticklabels(COLUMNS)

    # Add a title
    title = os.path.splitext(os.path.basename(out_png))[0]
    ax.set_title(f"Correlation Matrix Heatmap ({title})")

    # Annotate cells
    # Use a contrasting text color depending on background intensity
    vmax = np.nanmax(np.abs(corr.values))
    for i in range(len(COLUMNS)):
        for j in range(len(COLUMNS)):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Optionally save the correlation matrix CSV
    if save_corr_csv:
        if corr_csv_path is None:
            corr_csv_path = os.path.splitext(out_png)[0] + ".csv"
        corr.to_csv(corr_csv_path, float_format="%.6f")

    # Print a quick PSNR correlation summary to console
    psnr_corr = corr["psnr"].drop(labels=["psnr"])
    print(f"\nSaved heatmap to: {out_png}")
    if save_corr_csv:
        print(f"Saved correlation matrix to: {corr_csv_path}")
    print("Top correlations with PSNR:")
    print(psnr_corr.sort_values(ascending=False).to_string())

# -----------------------------
# Example usage (uncomment and edit paths as needed):
# make_corr_heatmap("candidates_bicycle.csv", "bicycle.png", save_corr_csv=True)
# make_corr_heatmap("candidates_bonsai.csv", "bonsai.png")
# make_corr_heatmap("candidates_counter.csv", "counter.png")
# make_corr_heatmap("candidates_flowers.csv", "flowers.png")
# make_corr_heatmap("candidates_garden.csv", "garden.png")
# make_corr_heatmap("candidates_kitchen.csv", "kitchen.png")
# make_corr_heatmap("candidates_room.csv", "room.png")
# make_corr_heatmap("candidates_stump.csv", "stump.png")
# -----------------------------
