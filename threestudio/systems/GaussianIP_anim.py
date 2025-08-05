from dataclasses import dataclass

import os
import torch
import torch.nn.functional as F

import numpy as np
import math
import random
from argparse import ArgumentParser
import yaml

import lpips

import threestudio
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.poser import Skeleton
from threestudio.utils.typing import *

from gaussiansplatting.gaussian_renderer import render, render_with_smaller_scale, render_deformed
from gaussiansplatting.arguments import PipelineParams, OptimizationParams
from gaussiansplatting.scene import GaussianModel, DeformedGaussianModel
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_model import BasicPointCloud

from utils.geometry import transformations
from utils.rotations import (
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)

from utils.human_body_prior.model_loader import load_model
from utils.human_body_prior.vposer_model import VPoser

from pytorch3d.ops.knn import knn_points

@threestudio.register("gaussianip-system")
class GaussianIP(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # basic settings
        log_path: str = "GaussianIP/logs"
        cur_time: str = ""
        config_path: str = "configs/exp.yaml"
        smplx_path: str = "/path/to/smplx/model"
        gender: str = 'neutral'
        stage: str = "stage1"
        apose: bool = True
        bg_white: bool = False
        ipa_ori: bool = True
        use_pose_controlnet: bool = True
        
        # 3dgs optimization settings
        pts_num: int = 100000
        sh_degree: int = 0
        height: int = 512
        width: int = 512
        ori_height: int = 1024
        ori_width: int = 1024
        head_offset: float = 0.65
        hand_radius: float = 0.05
        disable_hand_densification: bool = False
        
        # densify & prune settings
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        densify_prune_min_opacity: float = 0.15
        densify_prune_screen_size_threshold: int = 20
        densify_prune_world_size_threshold: float = 0.008
        densify_prune_screen_size_threshold_fix_step: int = 1500
        max_grad: float = 0.0002
        
        # prune_only settings
        prune_only_start_step: int = 1700
        prune_only_end_step: int = 1900
        prune_only_interval: int = 300
        prune_opacity_threshold: float = 0.05
        prune_screen_size_threshold: int = 20
        prune_world_size_threshold: float = 0.008

        # refine settings
        refine_start_step: int = 2400
        refine_n_views: int = 64
        refine_train_bs: int = 16
        refine_elevation: float = 17.
        refine_fovy_deg: float = 70.
        refine_camera_distance: float = 1.5
        refine_patch_size: int = 200
        refine_num_bboxes: int = 3
        lambda_l1: float = 1.0
        lambda_lpips: float = 0.5

        # anim settings
        vposer_path: str = "V02_05"
        anim_start_step: int = 3200
        use_non_rigid_position_delta: bool = True
        use_non_rigid_scale_delta: bool = True
        use_non_rigid_rotation_delta: bool = False
        delta_positions_weight: float = 0.01
        delta_scales_weight: float = 0.01

    cfg: Config

    def configure(self) -> None:
        # log related
        self.log_path = self.cfg.log_path
        self.cur_time = self.cfg.cur_time
        self.config_path = self.cfg.config_path
        self.stage = self.cfg.stage

        # 3dgs related
        self.sh_degree = self.cfg.sh_degree
        self.cameras_extent = 4.0
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.ipa_ori = self.cfg.ipa_ori
        self.use_pose_controlnet = self.cfg.use_pose_controlnet
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.head_offset = self.cfg.head_offset
        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        self.deformed_gaussian = DeformedGaussianModel()

        # refine related
        self.refine_start_step = self.cfg.refine_start_step
        self.refine_n_views = self.cfg.refine_n_views
        self.refine_train_bs = self.cfg.refine_train_bs
        self.refine_elevation = self.cfg.refine_elevation
        self.refine_fovy_deg = self.cfg.refine_fovy_deg
        self.refine_camera_distance = self.cfg.refine_camera_distance
        self.refine_patch_size = self.cfg.refine_patch_size
        self.refine_num_bboxes = self.cfg.refine_num_bboxes
        self.refine_batch = self.create_refine_batch()
        self.l1_loss_fn = F.l1_loss
        self.lpips_loss_fn = lpips.LPIPS(net='vgg')
        self.refine_loss = {'training_step': [], 'l1_loss': [], 'lpips_loss': []}
        self.refine_logger = []

        # animate related
        self.anim_start_step = self.cfg.anim_start_step
        self.vposer_path = self.cfg.vposer_path
        self.vposer, self.ps = load_model(self.vposer_path, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        self.vposer = self.vposer.to('cuda')
        self.use_non_rigid_position_delta = self.cfg.use_non_rigid_position_delta
        self.use_non_rigid_scale_delta = self.cfg.use_non_rigid_scale_delta
        self.use_non_rigid_rotation_delta = self.cfg.use_non_rigid_rotation_delta
        self.delta_positions_weight = self.cfg.delta_positions_weight
        self.delta_scales_weight = self.cfg.delta_scales_weight
        self.anim_test_batch = self.create_anim_batch(phase='test')

        # skeleton related
        self.skel = Skeleton(smplx_path=self.cfg.smplx_path, gender=self.cfg.gender, style="openpose", apose=self.cfg.apose)

        if self.stage != "stage4":
            self.skel.forward_smplx()
            self.skel.scale(-10)
        # NOTE: 在stage4中取消apose设定
        else:
            self.skel.apose = False
            self.get_A_verts()

    @torch.no_grad()
    def get_A_verts(self):
        a_body_pose = np.zeros((21, 3), dtype=np.float32)
        a_body_pose[0, 1] = 0.2
        a_body_pose[0, 2] = 0.1
        a_body_pose[1, 1] = -0.2
        a_body_pose[1, 2] = -0.1
        a_body_pose[15, 2] = -0.7853982
        a_body_pose[16, 2] = 0.7853982
        a_body_pose[19, 0] = 1.0
        a_body_pose[20, 0] = 1.0

        betas = torch.zeros([1, 10])
        self.betas = torch.tensor(betas, dtype=torch.float32, device='cuda')
        expression = torch.zeros([1, 10])
        self.expression = torch.tensor(expression, dtype=torch.float32, device='cuda')
        smplx_output = self.skel.smplx_model(
            body_pose=torch.tensor(a_body_pose, dtype=torch.float32, device='cuda').unsqueeze(0),
            betas=self.betas,
            expression=self.expression,
            return_verts=True
        )
        
        a_verts = smplx_output.vertices[0]
        self.T_t2a = smplx_output.T[0].detach()
        self.inv_T_t2a = torch.inverse(self.T_t2a)
        
        self.canonical_offsets = smplx_output.shape_offsets + smplx_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0]
        self.a_verts = a_verts.detach()
        return a_verts.detach()

    def pcd(self):
        points = self.skel.sample_smplx_points(N=self.cfg.pts_num)
        colors = np.ones_like(points) * 0.5
        pcd = BasicPointCloud(points, colors, None)
        return pcd

    def forward(self, batch: Dict[str, Any], renderbackground=None, phase='train') -> Dict[str, Any]:
        '''
        Render RGB images from self.gaussian given a batch of camera params, used in stage 1.
        '''
        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        all_vis_all = []
        self.viewspace_point_list = []

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w = batch['c2w'][id], FoVy = batch['fovy'][id], height = batch['height'], width = batch['width'])
            if phase == 'val' or phase == 'test':
                # render_pkg = render_with_smaller_scale(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            else:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)

            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.ipa_ori:
                enable_occlusion = True
                head_zoom = (batch['center'][id] == self.head_offset) & (batch['azimuth'][id] > 0)
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()
                azimuth = batch['azimuth'][id]
                
                if phase == 'train':
                    self.height = self.cfg.height
                    self.width = self.cfg.width
                else:
                    self.height = self.cfg.ori_height
                    self.width = self.cfg.ori_width

                if self.skel.style == 'humansd':
                    pose_image, _ = self.skel.humansd_draw(mvp, self.height, self.width, enable_occlusion)
                else:
                    pose_image, all_vis, _ = self.skel.openpose_draw(mvp, self.height, self.width, azimuth, head_zoom, enable_occlusion)

                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                all_vis_all.append(all_vis)
                pose_images.append(pose_image)

            else:
                enable_occlusion = True
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy() # [4, 4]
                pose_image, _ = self.skel.draw(mvp, self.height, self.width, enable_occlusion = True)
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                pose_images.append(pose_image)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)
        all_vis_all = torch.tensor(all_vis_all, device='cuda')

        self.visibility_filter = self.radii > 0.0

        # pass
        if self.cfg.disable_hand_densification:
            points = self.gaussian.get_xyz # [N, 3]
            hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to('cuda') # [2, 3]
            distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
            hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
            self.visibility_filter = self.visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg['all_vis_all'] = all_vis_all
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["scale"] = self.gaussian.get_scaling

        return {
            **render_pkg,
        }

    def create_refine_batch(self):
        '''
        Create a batch of camera parameters for VCR, used in stage 2,3.
        '''
        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = torch.linspace(-180., 180.0, self.refine_n_views + 1)[: self.refine_n_views]
        azimuth = azimuth_deg * math.pi / 180

        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        elevation_deg = torch.full_like(azimuth_deg, self.refine_elevation)
        elevation = elevation_deg * math.pi / 180

        camera_distances: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.refine_camera_distance)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),  # x
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),  # y
                camera_distances * torch.sin(elevation),                       # z
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(1, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.refine_fovy_deg)
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
        c2w: Float[Tensor, "B 4 4"] = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.width / self.height, 0.1, 1000.0)  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "mvp_mtx": mvp_mtx,
            "center": center[:,2],
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "height": self.cfg.ori_height,
            "width": self.cfg.ori_width,
            "fovy":fovy
        }
    
    def create_anim_batch(self, phase='test'):
        '''
        Create a batch of camera parameters for animation, used in stage 4.
        '''
        if phase == 'test':
            num_views = 180
        elif phase == 'train':
            num_views = 64

        elevation_deg = 0 # 17
        camera_distance = 1.5
        fovy_deg = 70
        znear = 0.1
        zfar = 1000.0

        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = torch.linspace(-180., 180.0, num_views + 1)[: num_views]
        azimuth = azimuth_deg * math.pi / 180

        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
        elevation = elevation_deg * math.pi / 180

        camera_distances: Float[Tensor, "B"] = torch.full_like(elevation_deg, camera_distance)

        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),  # x
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),  # y
                camera_distances * torch.sin(elevation),                       # z
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(1, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, fovy_deg)
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
        c2w: Float[Tensor, "B 4 4"] = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.width / self.height, znear, zfar)  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "mvp_mtx": mvp_mtx,
            "center": center[:,2],
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "height": self.cfg.ori_height,
            "width": self.cfg.ori_width,
            "fovy":fovy
        }

    def render_refine_rgb(self, phase='init', renderbackground=None) -> Dict[str, Any]:
        '''
        Render RGB images from self.gaussian given a batch of camera params, used in stage 3.
        '''
        if renderbackground is None:
            renderbackground = self.background_tensor
        
        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        assert phase in ['init', 'random', 'debug']

        if phase == 'init':
            id_list = [i for i in range(self.refine_n_views)]
        elif phase == 'random':
            id_list = random.sample(range(self.refine_n_views), self.refine_train_bs)
        else:
            id_list = [0, 8, 16, 24, 31]
        
        refine_height = self.refine_batch['height']
        refine_width = self.refine_batch['width']

        for idx, id in enumerate(id_list):
            viewpoint_cam = Camera(c2w = self.refine_batch['c2w'][id], FoVy = self.refine_batch['fovy'][id], height = refine_height, width = refine_width)
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # manually accumulate max radii across self.refine_batch
            if idx == 0:
                self.refine_radii = radii
            else:
                self.refine_radii = torch.max(radii, self.refine_radii)
                
            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)

            images.append(image) # [1024, 1024, 3]
            depths.append(depth) # [1024, 1024, 3]

            enable_occlusion = True
            head_zoom = (self.refine_batch['center'][id] == self.head_offset) & (self.refine_batch['azimuth'][id] > 0)
            mvp = self.refine_batch['mvp_mtx'][id].detach().cpu().numpy()
            azimuth = self.refine_batch['azimuth'][id]

            if self.skel.style == 'humansd':
                pose_image, _ = self.skel.humansd_draw(mvp, refine_height, refine_width, enable_occlusion)
            else:
                pose_image, _, _ = self.skel.openpose_draw(mvp, refine_height, refine_width, azimuth, head_zoom, enable_occlusion)

            pose_image = torch.from_numpy(pose_image).to('cuda')
            pose_images.append(pose_image) # [1024, 1024, 3]

        images = torch.stack(images, 0)             # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        depths = torch.stack(depths, 0)             # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        pose_images = torch.stack(pose_images, 0)   # [refine_n_views or refine_train_bs, 1024, 1024, 3]
        
        self.refine_visibility_filter = self.refine_radii > 0.0

        render_pkg["comp_rgb"] = images   
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images

        return {**render_pkg}, id_list

    def render_anim_rgb(self, batch: Dict[str, Any], phase='train', renderbackground=None) -> Dict[str, Any]:
        '''
        Render RGB images from deformed gaussian given a batch of camera params, used in stage 4.
        '''
        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        all_vis_all = []
        self.viewspace_point_list = []

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w = batch['c2w'][id], FoVy = batch['fovy'][id], height = batch['height'], width = batch['width'])
            # 获取deformed_gaussian的属性
            feats = self.deformed_gaussian.sh_features
            means3D = self.deformed_gaussian.xyz
            opacity = self.deformed_gaussian.opacity
            scales = self.deformed_gaussian.scaling
            rotations = self.deformed_gaussian.rotation
            active_sh_degree = self.sh_degree
            # TODO: 需检查相机参数
            render_pkg = render_deformed(viewpoint_cam, feats, means3D, opacity, scales, rotations, active_sh_degree, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # manually accumulate max radii across anim batch
            if id == 0:
                self.anim_radii = radii
            else:
                self.anim_radii = torch.max(radii, self.anim_radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.ipa_ori:
                enable_occlusion = True
                head_zoom = (batch['center'][id] == self.head_offset) & (batch['azimuth'][id] > 0)
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()
                azimuth = batch['azimuth'][id]
                
                # train的时候pose map是512, eval/test的时候是1024
                if phase == 'train':
                    self.height = self.cfg.height
                    self.width = self.cfg.width
                else:
                    self.height = self.cfg.ori_height
                    self.width = self.cfg.ori_width

                if self.skel.style == 'humansd':
                    pose_image, _ = self.skel.humansd_draw(mvp, self.height, self.width, enable_occlusion)
                else:
                    pose_image, all_vis, _ = self.skel.openpose_draw(mvp, self.height, self.width, azimuth, head_zoom, enable_occlusion)

                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                all_vis_all.append(all_vis)
                pose_images.append(pose_image)

            else:
                enable_occlusion = True
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy() # [4, 4]
                pose_image, _ = self.skel.draw(mvp, self.height, self.width, enable_occlusion = True)
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                pose_images.append(pose_image)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)
        all_vis_all = torch.tensor(all_vis_all, device='cuda')

        self.anim_visibility_filter = self.anim_radii > 0.0

        # pass
        if self.cfg.disable_hand_densification:
            points = self.gaussian.get_xyz # [N, 3]
            hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to('cuda') # [2, 3]
            distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
            hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
            self.anim_visibility_filter = self.anim_visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg['all_vis_all'] = all_vis_all
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["scale"] = self.gaussian.get_scaling

        return {
            **render_pkg,
        }

    def non_rigid_transform(self, delta_positions, delta_scales, delta_rotation) -> DeformedGaussianModel:
        '''
        Apply non-rigid transformation to canonical gaussian.
        '''
        # NOTE: 对position计算偏移量, 这里是直接在activate之后的值上进行偏移
        if self.use_non_rigid_position_delta:
            self.deformed_gaussian.xyz = self.gaussian.get_xyz + self.delta_positions_weight * delta_positions
        else:
            self.deformed_gaussian.xyz = self.gaussian.get_xyz

        # NOTE: 对scale计算偏移量, 这里是直接在activate之后的值上进行偏移(考虑到render函数里面也要先取激活值再渲染)
        if self.use_non_rigid_scale_delta:
            self.deformed_gaussian.scaling = self.gaussian.get_scaling + self.delta_scales_weight * delta_scales
        else:
            self.deformed_gaussian.scaling = self.gaussian.get_scaling

        # NOTE: 对rotation计算偏移量, 这里暂时不进行偏移
        if self.use_non_rigid_rotation_delta:
            self.deformed_gaussian.rotation = self.gaussian.get_rotation + delta_rotation
        else:
            self.deformed_gaussian.rotation = self.gaussian.get_rotation

        # deformed_gaussian的其他属性直接复制gaussian的
        self.deformed_gaussian.sh_features = self.gaussian.get_features
        self.deformed_gaussian.opacity = self.gaussian.get_opacity

        return self.deformed_gaussian

    def batch_index_select(self, data, inds):
        bs, nv = data.shape[:2]
        device = data.device
        inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        data = data.reshape(bs*nv, *data.shape[2:])
        return data[inds.long()]

    def smplx_lbs_diffuse_gau_topk(self, lbs_weights, verts_transform, points, template_points, K):
        '''
        ref: https://github.com/JanaldoChen/Anim-NeRF
        '''
        bz, np, _ = points.shape
        with torch.no_grad():
            results = knn_points(points, template_points, K=K)
            dists, idxs = results.dists, results.idx

        NNs_dist = dists
        NNs = idxs
        weight_std = 0.1
        weight_std2 = 2. * weight_std ** 2
        conf_threshold = 0.0

        # 取出各高斯点的最近邻, [1, gau_num, k, joint_num]
        gau_NNs_lbs_weights = lbs_weights[NNs]  
        
        # 计算各最近邻的置信度, [1, gau_num, k]
        gau_NNs_conf = torch.exp(
            -torch.sum(
                torch.abs(gau_NNs_lbs_weights - gau_NNs_lbs_weights[..., 0:1, :]), dim=-1
            ) / weight_std2
        )
        # 卡阈值筛选出置信度高的最近邻顶点, 只有这些顶点贡献权重
        gau_NNs_conf = torch.gt(gau_NNs_conf, conf_threshold).float() # [1, gau_num, k], 0或1
        # 映射到(0, 1)区间
        gau_NNs_weights = torch.exp(-NNs_dist)  # [1, gau_num, k]
        # 乘上置信度mask
        gau_NNs_weights *= gau_NNs_conf         # [1, gau_num, k]
        # 影响因子归一化
        gau_NNs_weights = gau_NNs_weights / gau_NNs_weights.sum(-1, keepdim=True)  # [1, gau_num, k]

        # 取出k个最近邻顶点的变换矩阵
        gau_NNs_transform = self.batch_index_select(verts_transform, NNs)  # [1, gau_num, k, 4, 4]
        # 计算这些变换矩阵的加权和, [1, gau_num, 4, 4]
        gau_transform = torch.sum(gau_NNs_weights.unsqueeze(-1).unsqueeze(-1) * gau_NNs_transform, dim=2)
        
        return gau_transform

    def rigid_transform(self, gaussian: DeformedGaussianModel) -> DeformedGaussianModel:
        '''
        Apply rigid transformation to non-rigid deformed gaussian.
        '''
        gs_xyz = gaussian.xyz
        gs_scaling = gaussian.scaling
        gs_opacity = gaussian.opacity
        gs_sh_features = gaussian.sh_features
        gs_rotq = gaussian.rotation

        # gs_scales_cano = gs_scaling.clone()
        gs_rotmat = quaternion_to_matrix(gs_rotq)

        # a-pose -> t-pose -> tgt-pose
        # remove & reapply the blendshape
        curr_offsets = (self.smplx_output.shape_offsets + self.smplx_output.pose_offsets)[0]
        T_t2pose = self.smplx_output.T[0]
        T_a2t = self.inv_T_t2a.clone()
        T_a2t[..., :3, 3] = T_a2t[..., :3, 3] + self.canonical_offsets - curr_offsets
        T_a2pose = T_t2pose @ T_a2t

        lbs_gau = self.smplx_lbs_diffuse_gau_topk(
            lbs_weights=self.skel.smplx_model.lbs_weights,  # 固定的smplx顶点的lbs权重
            verts_transform=T_a2pose.unsqueeze(0),          # 所有smplx顶点的a2pose变换
            points=self.gaussian._xyz.unsqueeze(0),         # NOTE: 应该用最初标准姿态下的高斯位置还是non-rigid变形后的高斯位置?
            template_points=self.a_verts.unsqueeze(0),      # 标准姿态下的mesh顶点位置
            K=16
        ).squeeze(0)

        homogen_coord = torch.ones_like(gs_xyz[..., :1])

        # apply rigid deform to xyz
        gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
        rigid_deformed_xyz = torch.matmul(lbs_gau, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        # apply rigid deform to rotation
        rigid_deformed_gs_rotmat = lbs_gau[:, :3, :3] @ gs_rotmat
        rigid_deformed_gs_rotq = matrix_to_quaternion(rigid_deformed_gs_rotmat)

        ext_tfs = None
        # manual_trans = torch.tensor((np.array([0, -1, 0])), dtype=torch.float32, device='cuda')
        # manual_rot = np.array([-90.0, 0, 0]) / 180 * np.pi
        # manual_rotmat = torch.tensor((transformations.euler_matrix(*manual_rot)[:3, :3]),dtype=torch.float32, device='cuda')
        # manual_scale = torch.tensor((1.0), dtype=torch.float32, device='cuda')
        # ext_tfs = (manual_trans.unsqueeze(0), manual_rotmat.unsqueeze(0), manual_scale.unsqueeze(0))

        if ext_tfs is not None:
            # NOTE: 用外部自定义的后处理方式对xyz, rot, scale进行最终处理, 防止渲染结果中的人物位置或镜头朝向不合理
            tr, rotmat, sc = ext_tfs
            rigid_deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ rigid_deformed_xyz[..., None]))).squeeze(-1)
            gs_scaling = sc * gs_scaling

            rotq = matrix_to_quaternion(rotmat)
            rigid_deformed_gs_rotq = quaternion_multiply(rotq, rigid_deformed_gs_rotq)
            rigid_deformed_gs_rotmat = quaternion_to_matrix(rigid_deformed_gs_rotq)

        self.deformed_gaussian.xyz = rigid_deformed_xyz.to(dtype=torch.float32)
        self.deformed_gaussian.rotation = rigid_deformed_gs_rotq.to(dtype=torch.float32)
        self.deformed_gaussian.scaling = gs_scaling.to(dtype=torch.float32)
        self.deformed_gaussian.opacity = gs_opacity
        self.deformed_gaussian.sh_features = gs_sh_features
        
        return self.deformed_gaussian

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # stage 1: AHDS training
        if self.stage == "stage1":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.prepare_for_sds(self.prompt_processor.prompt, self.prompt_processor.negative_prompt, self.prompt_processor.null_prompt, self.stage)
        # stage 3: 3d reconstruction
        elif self.stage == "stage3":
            self.refined_rgbs_small = torch.load(os.path.join(self.log_path, self.cur_time, 'after_refine.pth'))['refined_rgbs_small'].to(self.device)
        # stage 4: animation
        elif self.stage == "stage4":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.prepare_for_sds(self.prompt_processor.prompt, self.prompt_processor.negative_prompt, self.prompt_processor.null_prompt, self.stage)

    def training_step(self, batch, batch_idx):
        if self.true_global_step < self.refine_start_step-1 and self.stage == "stage1":
            self.gaussian.update_learning_rate(self.true_global_step)

            out = self(batch, phase='train')

            prompt_utils = self.prompt_processor()
            images = out["comp_rgb"]
            control_images = out['pose']
            all_vis_all = out['all_vis_all']

            guidance_out = self.guidance(self.true_global_step, images, control_images, prompt_utils, self.use_pose_controlnet, all_vis_all, **batch)
            
            # init loss
            loss = 0.0

            # loss_sds
            loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])
            
            # loss_sparsity
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            
            # loss_opaque
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

            return {"loss": loss}

        elif self.true_global_step == self.refine_start_step-1 and self.stage == "stage1":
            gs_out, _ = self.render_refine_rgb(phase='init', renderbackground=None)
            images = gs_out["comp_rgb"].detach()      # [refine_n_views, H, W, 3]
            control_images = gs_out['pose'].detach()  # [refine_n_views, H, W, 3]
            # save image data before refine
            images = images.to('cpu')
            control_images = control_images.to('cpu')
            torch.save({'images': images, 'control_images': control_images}, os.path.join(self.log_path, self.cur_time, 'before_refine.pth'))

            # self.refined_rgbs, self.view_idx_all = self.guidance.refine_rgb(images, control_images, self.prompt_processor.prompt)  # [refine_n_views, H, W, 3]
            
            self.view_idx_all = [24, 8, 16, 0, 20, 28, 4, 12, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            # save images before refine
            for i, _ in enumerate(self.view_idx_all):
                cur_raw_rgb = images[i]
                cur_control_image = control_images[i]
                # cur_refined_rgb = self.refined_rgbs[i]
                self.save_image(f"raw_rgb_{i}.png", cur_raw_rgb)
                self.save_image(f"control_image_{i}.png", cur_control_image)
                # self.save_image(f"refined_rgb_{view_idx}.png", cur_refined_rgb)

            # self.idx_mapper = [3, 20, 21, 22, 6, 23, 24, 25, 1, 26, 27, 28, 7, 29, 30, 31, 2, 8, 9, 10, 4, 11, 12, 13, 0, 14, 15, 16, 5, 17, 18, 19]
            # self.refined_rgbs = self.refined_rgbs[self.idx_mapper]
            # self.refined_rgbs = self.refined_rgbs.permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
            # self.refined_rgbs_small = F.interpolate(self.refined_rgbs, scale_factor=0.5, mode="bilinear", align_corners=False)
            return None
        
        elif self.stage == "stage3":
            self.gaussian.update_learning_rate(self.true_global_step + self.refine_start_step)
            gs_out, id_list = self.render_refine_rgb(phase='random', renderbackground=None)
            rgb_render = gs_out["comp_rgb"].permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
            rgb_render_small = F.interpolate(rgb_render, scale_factor=0.5, mode="bilinear", align_corners=False)
            rgb_gt_small = self.refined_rgbs_small[id_list]

            # init loss
            loss = 0.0
            # l1 and lpips loss, use crop and downsample to save memory
            l1_loss = self.l1_loss_fn(rgb_render_small, rgb_gt_small)
            lpips_loss = self.lpips_loss_fn(rgb_render_small, rgb_gt_small, normalize=True).mean()
            loss = loss + self.cfg.lambda_l1 * l1_loss + self.cfg.lambda_lpips * lpips_loss

            # record loss
            self.refine_loss['training_step'].append(self.true_global_step)
            self.refine_loss['l1_loss'].append(l1_loss.item())
            self.refine_loss['lpips_loss'].append(lpips_loss.item())

            return {"loss": loss}

        elif self.stage == "stage4":
            # TODO: deform_network和各高斯属性的lr是否合理
            self.gaussian.update_deform_learning_rate(self.true_global_step + self.anim_start_step)
            self.gaussian.update_learning_rate(self.true_global_step + self.anim_start_step)

            # NOTE: 1. 通过V-Poser随机采样一个姿态参数
            latent_pose_code = torch.from_numpy(np.random.randn(1, 32).astype(np.float32)).to('cuda')
            with torch.no_grad():
                body_pose = self.vposer.decode(latent_pose_code)['pose_body'].contiguous().view(-1, 63)

            # NOTE: 每执行一遍forward_smplx都会得到新的self.skel.points3D用于绘制pose map
            # NOTE: 似乎需要传入gloabl_orient
            self.smplx_output = self.skel.forward_smplx(body_pose=body_pose)
            self.skel.scale(-10)

            # TODO: 2. 对canonical space中的高斯进行非刚性形变
            delta_positions, delta_scales, delta_quaternions = self.gaussian.deform_model(x=self.gaussian.get_xyz, body_pose=body_pose)
            self.deformed_gaussian = self.non_rigid_transform(delta_positions, delta_scales, delta_quaternions)

            # TODO: 3. 对非刚性形变后的高斯进行刚性形变
            # NOTE: 理论上而言，只要用anime中的相机参数去渲染，就能得到正确的结果
            self.deformed_gaussian = self.rigid_transform(self.deformed_gaussian)

            # TODO: 4. render_anim_rgb中需要对**形变后的高斯**用batch中的视角进行渲染
            out = self.render_anim_rgb(batch, phase='train')

            prompt_utils = self.prompt_processor()
            images = out["comp_rgb"]
            control_images = out['pose']
            all_vis_all = out['all_vis_all']

            guidance_out = self.guidance(self.true_global_step, images, control_images, prompt_utils, self.use_pose_controlnet, all_vis_all, **batch)
            
            # init loss
            loss = 0.0

            # loss_sds
            loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])
            
            # loss_sparsity
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            
            # loss_opaque
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

            return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):
        if self.true_global_step % 100 == 0:
            threestudio.info('Gaussian points num: {}'.format(self.gaussian.get_features.shape[0]))
        if self.true_global_step < self.refine_start_step and self.stage == "stage1":
            with torch.no_grad():
                if self.true_global_step < self.cfg.densify_prune_end_step:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                    # densify_and_prune
                    self.min_opacity = self.cfg.densify_prune_min_opacity if self.true_global_step > 1900 else 0.05
                    if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:
                        densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
                        self.gaussian.densify_and_prune(self.cfg.max_grad, self.min_opacity, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

                # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
                if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                    if self.true_global_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)
        
        elif self.stage == "stage3":
            with torch.no_grad():
                if self.true_global_step + self.refine_start_step < 10000:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    if self.true_global_step == 0:
                        # When stage 3 starts, the loaded gaussians don't have max_radii2D
                        self.gaussian.max_radii2D = self.refine_radii  # self.get_xyz.shape[0]
                        self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    else:
                        self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    # densify_and_prune
                    if self.true_global_step + self.refine_start_step == 2500:
                        densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
                        self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

                # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
                if self.true_global_step + self.refine_start_step > 2500 and self.true_global_step + self.refine_start_step < 3000:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    # 2700的时候prune_only一次
                    if self.true_global_step + self.refine_start_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)

        # TODO: stage4的densify和prune策略
        elif self.stage == "stage4":
            with torch.no_grad():
                if self.true_global_step + self.anim_start_step < 10000:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    if self.true_global_step == 0:
                        # When stage 4 starts, the loaded gaussians don't have max_radii2D
                        self.gaussian.max_radii2D = self.anim_radii
                        self.gaussian.max_radii2D[self.anim_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.anim_visibility_filter], self.anim_radii[self.anim_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.anim_visibility_filter)
                    else:
                        self.gaussian.max_radii2D[self.anim_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.anim_visibility_filter], self.anim_radii[self.anim_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.anim_visibility_filter)
                    # NOTE: 动态建模阶段暂时不densify
                    # if self.true_global_step + self.refine_start_step == 2500:
                    #     densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
                    #     self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

                # do prune-only once
                if self.true_global_step + self.anim_start_step > 4700 and self.true_global_step + self.anim_start_step < 4900:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.anim_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.anim_visibility_filter], self.anim_radii[self.anim_visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.anim_visibility_filter)
                    if self.true_global_step + self.anim_start_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)

    def validation_step(self, batch, batch_idx):
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]
        background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 新创建的tensor需指定device
        
        out = self.render_anim_rgb(batch, phase='val')
        if self.stage == "stage1":
            out = self(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])
        elif self.stage == "stage3":
            out = self(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])
        elif self.stage == "stage4":
            # NOTE: 渲染视角可能有问题
            out = self.render_anim_rgb(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step + self.anim_start_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]
        background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 新创建的tensor需指定device
        if self.stage == "stage1":
            out = self(batch, renderbackground=background_tensor, phase='test')
            if only_rgb:
                self.save_image(f"it{self.true_global_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
                self.save_image(f"it{self.true_global_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])

        elif self.stage == "stage3":
            out = self(batch, renderbackground=background_tensor, phase='test')
            if only_rgb:
                self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
                self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])
        
        elif self.stage == "stage4":
            # out = self(self.anim_test_batch, renderbackground=background_tensor, phase='test')
            out = self.render_anim_rgb(self.anim_test_batch, renderbackground=background_tensor, phase='test')
            images = out["comp_rgb"].detach()      # [refine_n_views, H, W, 3]
            control_images = out['pose'].detach()  # [refine_n_views, H, W, 3]
            images = images.to('cpu')
            control_images = control_images.to('cpu')

            for i in range(images.shape[0]):
                cur_raw_rgb = images[i]
                cur_control_image = control_images[i]
                # cur_refined_rgb = self.refined_rgbs[i]
                self.save_image(f"it{self.true_global_step + self.anim_start_step}-test/rgb/{i}.png", cur_raw_rgb)
                self.save_image(f"it{self.true_global_step + self.anim_start_step}-test/pose/{i}.png", cur_control_image)

            # out = self.render_anim_rgb(batch, renderbackground=background_tensor, phase='test')
            # out = self.render_anim_rgb(self.anim_test_batch, renderbackground=background_tensor, phase='test')
            # if only_rgb:
                # self.save_image(f"it{self.true_global_step + self.anim_start_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
            #     self.save_image(f"it{self.true_global_step + self.anim_start_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])

    def on_test_epoch_end(self):
        if self.stage == "stage1":
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"cano_after_stage1.ply"))
        
        elif self.stage == "stage3":
            # save mp4
            self.save_img_sequence(
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step + self.refine_start_step,
            )

            # save 3dgs ply
            save_path = self.get_save_path(f"it{self.true_global_step + self.refine_start_step}-test/cano_after_stage3.ply")
            self.gaussian.save_ply(save_path)
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"cano_after_stage3.ply"))

            # change the max_steps in config.yaml from total_step to refine_start_step
            config_file_path = self.config_path

            # read config.yaml
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)

            # change args
            config['system']['stage'] = 'stage1'
            config['trainer']['max_steps'] = self.refine_start_step

            # write it back to config.yaml
            with open(config_file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            print(f"Updated max_steps to {self.refine_start_step} in {config_file_path}.")

        elif self.stage == "stage4":
            # save mp4
            self.save_img_sequence(
                f"it{self.true_global_step + self.anim_start_step}-test/rgb",
                f"it{self.true_global_step + self.anim_start_step}-test/rgb",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step + self.anim_start_step,
            )

            # save 3dgs ply
            save_path = self.get_save_path(f"it{self.true_global_step + self.anim_start_step}-test/cano_after_stage4.ply")
            self.gaussian.save_ply(save_path)
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"cano_after_stage4.ply"))

            # config_file_path = self.config_path

            # read config.yaml
            # with open(config_file_path, 'r') as file:
            #     config = yaml.safe_load(file)

            # change args
            # config['system']['stage'] = 'stage1'
            # config['trainer']['max_steps'] = self.refine_start_step

            # write it back to config.yaml
            # with open(config_file_path, 'w') as file:
            #     yaml.dump(config, file, default_flow_style=False)

            # print(f"Updated max_steps to {self.refine_start_step} in {config_file_path}.")
            
    def configure_optimizers(self):
        if self.stage == "stage1":
            opt = OptimizationParams(self.parser)
            point_cloud = self.pcd()
            self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
            self.gaussian.training_setup(opt, self.stage)
            ret = {"optimizer": self.gaussian.optimizer}
        elif self.stage == "stage3":
            # load 3dgs from stage 1 output
            opt = OptimizationParams(self.parser)
            # NOTE: ！！！！！！用的是sorted！！！！！！应该是为了适配训练的时候用的相机参数
            # NOTE: ！！！！！！animation.py用的是无sorted！！！！！！应该是为了适配animation的时候用的相机参数
            self.gaussian.load_ply(os.path.join(self.log_path, self.cur_time, f"cano_after_stage1.ply"))
            self.gaussian.training_setup(opt, self.stage)
            ret = {"optimizer": self.gaussian.optimizer}
        elif self.stage == "stage4":
            # load 3dgs from stage 3 output
            opt = OptimizationParams(self.parser)
            # TODO: 最后改为加载某个trail中的结果
            # NOTE: ！load ply方式有差异！
            # self.gaussian.load_ply(os.path.join("/root/autodl-tmp/GaussianIP/data/humans/last.ply"))
            self.gaussian.load_ply_anim(os.path.join("/root/autodl-tmp/GaussianIP/data/humans/last.ply"))
            # self.gaussian.load_ply_anim(os.path.join(self.log_path, self.cur_time, f"cano_after_stage3.ply"))
            # 设置各参数的学习率, 配置优化器
            self.gaussian.training_setup(opt, self.stage)
            ret = {"optimizer": self.gaussian.optimizer}
        return ret
