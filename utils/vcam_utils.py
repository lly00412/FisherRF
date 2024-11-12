from scene.cameras import VirtualCam
from utils.graphics_utils import getIntrinsicMatrix
from utils.proj_utils import *
from utils.uncert_utils import *
from gaussian_renderer import modified_render
from typing import Optional, Tuple
from torch import nn
import torch

def render_vcam_difference(render_pkg, view, gaussians, pipeline, background, n_vcam=6,r_scale=0.1,method='vcam'):
    depth = render_pkg['depth'].squeeze()
    pred_img = render_pkg['render']
    look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
    D_median = depth.clone().flatten().median(0).values
    # radiaus = 0.1*D_median

    rd_c2w = rd_c2w.to(depth.device)
    K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                           fovX=view.FoVx, fovY=view.FoVy).to(depth.device)  # (4,4)
    GetVcam = VirtualCam(view)
    backwarp = BackwardWarping(out_hw=(view.image_height, view.image_width),
                               device=depth.device, K=K)

    # random sampling n virtual camera at a sphere centering at real camera
    radiaus = r_scale * D_median
    Vcams = GetVcam.get_N_near_cam_by_look_at(n_vcam, look_at=look_at, radiaus=radiaus)

    rd_depth = depth.unsqueeze(0).unsqueeze(0)
    rd_depths = rd_depth.repeat(n_vcam, 1, 1, 1)
    rd_pred_imgs = pred_img.unsqueeze(0).repeat(n_vcam, 1, 1, 1)

    vir_depths = []
    vir_pred_imgs = []
    rd2virs = []
    for vir_view in Vcams:
        vir_render_pkg = modified_render(vir_view, gaussians, pipeline, background)
        vir_depth = vir_render_pkg['depth'].squeeze()
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
    # nv_mask ()
    vir2rd_depths[nv_mask.bool()] = 0.
    nv_mask = nv_mask.repeat(1,3,1,1)
    vir2rd_pred_imgs[nv_mask.bool()] = 0.
    depth_l2 = (vir2rd_depths - rd_depths) **2
    rgb_l2 = (vir2rd_pred_imgs - rd_pred_imgs) **2
    diff = torch.cat([depth_l2,rgb_l2],dim=1)
    return diff, nv_mask

# refer to https://github.com/AaltoML/uncertainty-nerf-gs/blob/main/nerfuncertainty/utils.py

def create_mlp(
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: int,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU,
        out_activation: nn.Module = None,
        dropout_layers: Optional[Tuple[float]] = None,
        dropout_rate: Optional[float] = None,
        dtype: torch.dtype = torch.float32
):
    layers = []
    skip_connections = set(skip_connections) if skip_connections else set()
    dropout_layers = set(dropout_layers) if dropout_layers else set()

    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))
    else:
        for i in range(num_layers - 1):
            if i in dropout_layers:
                layers.append(nn.Dropout(p=dropout_rate))
            if i == 0:
                assert i not in skip_connections, "No skip connection for layer 0"
                layers.append(nn.Linear(in_dim, layer_width, dtype=dtype))
            elif i in skip_connections:
                layers.append(nn.Linear(in_dim + layer_width, layer_width, dtype=dtype))
            else:
                layers.append(nn.Linear(layer_width, layer_width, dtype=dtype))
            if activation:
                layers.append(activation())
        if (num_layers - 1) in dropout_layers or (-1 in dropout_layers):
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(layer_width, out_dim, dtype=dtype))
        if out_activation:
            layers.append(out_activation())
    return nn.Sequential(*layers)