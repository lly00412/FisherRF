#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,getView2World

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class VirtualCam(nn.Module):
    def __init__(self, gt_cam, data_device="cuda"
                 ):
        super(VirtualCam, self).__init__()
        self.gt_cam = gt_cam
        self.data_device = self.gt_cam.data_device
        self.get_camera_direction_and_center()

    def get_camera_direction_and_center(self):
        # self.camera_center = self.gt_cam.camera_center.clone()  # torch.tensor
        # w2c = self.gt_cam.world_view_transform.T
        # c2w = torch.linalg.inv(w2c)
        c2w = torch.tensor(getView2World(self.gt_cam.R, self.gt_cam.T))
        self.camera_center = c2w[:3,3].clone().to(self.data_device)
        self.left = c2w[:3, 0].clone().to(self.data_device)
        self.up = c2w[:3, 1].clone().to(self.data_device)
        self.forward = c2w[:3, 2].clone().to(self.data_device)

    def get_rotation_matrix(self, theta=5, axis='x'):  # rot theta degree across x axis
        phi = (theta * (np.pi / 180.))
        rot = torch.eye(4)
        if axis == 'x':
            rot[:3, :3] = torch.Tensor([
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)]
            ])
        elif axis == 'y':
            rot[:3, :3] = torch.Tensor([
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)]
            ])
        elif axis == 'z':
            rot[:3, :3] = torch.Tensor([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ])
        return rot

    def get_rotation_by_direction(self, theta=5, direction='u'):
        if direction == 'u':
            theta = 0 - theta
            rot = self.get_rotation_matrix(theta, axis='x')
        elif direction == 'd':
            rot = self.get_rotation_matrix(theta, axis='x')
        elif direction == 'l':
            theta = 0 - theta
            rot = self.get_rotation_matrix(theta, axis='y')
        elif direction == 'r':
            rot = self.get_rotation_matrix(theta, axis='y')
        elif direction == 'f':
            theta = 0
            rot = self.get_rotation_matrix(theta, axis='y')
        elif direction == 'b':
            theta = 180
            rot = self.get_rotation_matrix(theta, axis='y')
        return rot.to(self.data_device)

    def get_translation_matrix(self, origin, destination):  # both should be (x,y,z)
        trans = torch.eye(4).to(destination)
        trans[:3, 3] = destination - origin
        return trans

    def get_near_cam_by_look_at(self, look_at, theta=3, direction='u',stepsize=0.1):

        # rotation the camera around scene center
        # move the camera forward/backward by stepsize
        trans_1 = self.get_translation_matrix(self.camera_center, look_at)
        rot = self.get_rotation_by_direction(theta, direction)

        unit_direction = (look_at-self.camera_center)/torch.norm(look_at-self.camera_center)
        random_step = (torch.rand(1).to(stepsize) * 2 * stepsize) - stepsize
        new_center = self.camera_center + random_step * unit_direction
        trans_2 = self.get_translation_matrix(look_at, new_center)

        # c2w_homo = torch.eye(4).to(self.data_device)
        # c2w_homo[:3] = self.gt_cam.world_view_transform.inverse()[:3].clone()
        w2c = self.gt_cam.world_view_transform.T.clone()
        # w2c = torch.inverse(trans_1) @ rot @ trans_1 @ w2c
        w2c = trans_2 @ rot @ trans_1 @ w2c

        world_view_transform = w2c.transpose(0, 1).to(self.data_device)
        projection_matrix = self.gt_cam.projection_matrix
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        VirtualCam = MiniCam(width=self.gt_cam.image_width, height=self.gt_cam.image_height,
                             fovy=self.gt_cam.FoVy, fovx=self.gt_cam.FoVx, znear=self.gt_cam.znear,
                             zfar=self.gt_cam.zfar,
                             world_view_transform=world_view_transform, full_proj_transform=full_proj_transform)
        return VirtualCam