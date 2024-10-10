'''
refer to https://github.com/oppo-us-research/NARUTO/tree/release/src/layers
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.graphics_utils import getIntrinsicMatrix,getView2World

class Projection(nn.Module):
    """Layer which projects 3D points into a camera view
    """
    def __init__(self, height, width, eps=1e-7):
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points3d, K, normalized=True):
        """
        Args:
            points3d (torch.tensor, [N,4,(HxW)]: 3D points in homogeneous coordinates
            K (torch.tensor, [torch.tensor, (N,4,4)]: camera intrinsics
            normalized (bool):
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            xy (torch.tensor, [N,H,W,2]): pixel coordinates
        """
        # projection
        points2d = torch.matmul(K[:, :3, :], points3d)

        # convert from homogeneous coordinates
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + self.eps)
        xy = xy.view(points3d.shape[0], 2, self.height, self.width)
        xy = xy.permute(0, 2, 3, 1)

        # normalization
        if normalized:
            xy[..., 0] /= self.width - 1
            xy[..., 1] /= self.height - 1
            xy = (xy - 0.5) * 2
        return xy

class Transformation3D(nn.Module):
    """Layer which transform 3D points
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self,
                points: torch.Tensor,
                T: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
            T (torch.Tensor, [N,4,4]): transformation matrice
        Returns:
            transformed_points (torch.Tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points

class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics

    Attributes
        xy (torch.tensor, [N,3,HxW]: homogeneous pixel coordinates on regular grid
    """

    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        self.xy = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
            , 0)
        self.xy = torch.cat([self.xy, self.ones], 1)
        self.xy = nn.Parameter(self.xy, requires_grad=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (torch.tensor, [N,1,H,W]: depth map
            inv_K (torch.tensor, [N,4,4]): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is [N,4,H,W]; else [N,4,(HxW)]
        Returns:
            points (torch.tensor, [N,4,(HxW)]): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1)
        ones = self.ones.repeat(depth.shape[0], 1, 1)

        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points

class BackwardWarping(nn.Module):

    def __init__(self,
                 out_hw: Tuple[int,int],
                 device: torch.device,
                 K:torch.Tensor) -> None:
        super(BackwardWarping,self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height,width).to(device)
        self.projection = Projection(height,width).to(device)
        self.transform3d = Transformation3D().to(device)

        H,W = height,width
        self.rgb = torch.zeros(H,W,3).view(-1,3).to(device)
        self.depth = torch.zeros(H, W, 1).view(-1, 1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K).to(device)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0) # 1,4,4
    def forward(self,
                img_src: torch.Tensor,
                depth_src: torch.Tensor,
                depth_tgt: torch.Tensor,
                tgt2src_transform: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = depth_tgt.shape

        # reproject
        pts3d_tgt = self.backproj(depth_tgt,self.inv_K)
        pts3d_src = self.transform3d(pts3d_tgt,tgt2src_transform)
        src_grid = self.projection(pts3d_src,self.K,normalized=True)
        transformed_distance = pts3d_src[:, 2:3].view(b,1,h,w)

        img_tgt = F.grid_sample(img_src, src_grid, mode = 'bilinear', padding_mode = 'zeros')
        depth_src2tgt = F.grid_sample(depth_src, src_grid, mode='bilinear', padding_mode='zeros')

        # rm invalid depth
        valid_depth_mask = (transformed_distance < 1e6) & (depth_src2tgt > 0)

        # rm invalid coords
        vaild_coord_mask = (src_grid[...,0]> -1) & (src_grid[...,0] < 1) & (src_grid[...,1]> -1) & (src_grid[...,1] < 1)
        vaild_coord_mask = vaild_coord_mask.unsqueeze(1)

        valid_mask = valid_depth_mask & vaild_coord_mask
        invaild_mask = ~valid_mask

        return img_tgt.float(), depth_src2tgt.float(), invaild_mask.float()

def extract_scene_center_and_C2W(depth, view):
    K = getIntrinsicMatrix(width=view.image_width, height=view.image_height, fovX=view.FoVx, fovY=view.FoVy).to(
        depth)
    inv_K = torch.inverse(K).unsqueeze(0)
    backproj_func = Backprojection(height=view.image_height, width=view.image_width)
    depth_v = depth.clone()
    depth_v = depth_v.unsqueeze(0).unsqueeze(0)
    mask = (depth_v < depth_v.max()).squeeze(0)
    point3d_camera = backproj_func(depth_v.cpu(), inv_K.cpu(), img_like_out=True).squeeze(0)
    # C2W = torch.tensor(getView2World(view.R, view.T))
    C2W = view.world_view_transform.transpose(0,1).inverse().cpu()
    point3d_world = C2W @ point3d_camera.view(4, -1)
    point3d_world = point3d_world.view(4, point3d_camera.shape[1], point3d_camera.shape[2])
    expanded_mask = mask.expand_as(point3d_world)
    selected = point3d_world.to(mask.device)[expanded_mask]
    selected = selected.view(4, -1)
    look_at = selected.median(1).values[:3]
    # look_at = selected.mean(1)[:3]
    return look_at,C2W