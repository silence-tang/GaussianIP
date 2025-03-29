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
import matplotlib.pyplot as plt

import threestudio
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.poser import Skeleton
from threestudio.utils.typing import *

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import PipelineParams, OptimizationParams
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_model import BasicPointCloud

    

@threestudio.register("gaussianip-system")
class GaussianIP(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # basic settings
        log_path: str = "GaussianIP"
        cur_time: str = ""
        config_path: str = "configs/exp.yaml"
        stage: str = "stage1"
        apose: bool = True
        bg_white: bool = False
        radius: float = 4
        ipa_ori: bool = True
        use_pose_controlnet: bool = False
        smplx_path: str = "/path/to/smplx/model"
        pts_num: int = 100000
        sh_degree: int = 0
        height: int = 512
        width: int = 512
        ori_height: int = 1024
        ori_width: int = 1024
        head_offset: float = 0.65

        # 3dgs optimization related
        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        
        # densify & prune settings
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        densify_prune_min_opacity: float = 0.15
        densify_prune_screen_size_threshold: int = 20
        densify_prune_world_size_threshold: float = 0.008
        densify_prune_screen_size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        gender: str = 'neutral'
        
        # prune_only settings
        prune_only_start_step: int = 1700
        prune_only_end_step: int = 1900
        prune_only_interval: int = 300
        prune_opacity_threshold: float = 0.05
        prune_screen_size_threshold: int = 20
        prune_world_size_threshold: float = 0.008

        # refine related
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

    cfg: Config

    def configure(self) -> None:
        self.log_path = self.cfg.log_path
        self.cur_time = self.cfg.cur_time
        self.config_path = self.cfg.config_path
        self.stage = self.cfg.stage
        self.radius = self.cfg.radius
        self.gaussian = GaussianModel(sh_degree = self.cfg.sh_degree)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.ipa_ori = self.cfg.ipa_ori
        self.use_pose_controlnet = self.cfg.use_pose_controlnet
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.head_offset = self.cfg.head_offset

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

        if self.ipa_ori:
            self.skel = Skeleton(style="openpose", apose=self.cfg.apose)
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        else:
            self.skel = Skeleton(apose=self.cfg.apose)
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        
        self.cameras_extent = 4.0


    def pcd(self):
        points = self.skel.sample_smplx_points(N=self.cfg.pts_num)
        colors = np.ones_like(points) * 0.5
        pcd = BasicPointCloud(points, colors, None)
        return pcd
    

    def forward(self, batch: Dict[str, Any], renderbackground=None, phase='train') -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        all_vis_all = []
        self.viewspace_point_list = []

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam  = Camera(c2w = batch['c2w'][id], FoVy = batch['fovy'][id], height = batch['height'], width = batch['width'])
            if phase == 'val' or phase == 'test':
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


    def render_refine_rgb(self, phase='init', renderbackground=None):
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
            viewpoint_cam  = Camera(c2w = self.refine_batch['c2w'][id], FoVy = self.refine_batch['fovy'][id], height = refine_height, width = refine_width)
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


    def on_fit_start(self) -> None:
        super().on_fit_start()
        # stage 1: AHDS training
        if self.stage == "stage1":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.prepare_for_sds(self.prompt_processor.prompt, self.prompt_processor.negative_prompt, self.prompt_processor.null_prompt)
        # stage 3: 3d reconstruction
        else:
            self.refined_rgbs_small = torch.load(os.path.join(self.log_path, self.cur_time, 'after_refine.pth'))['refined_rgbs_small'].to(self.device)


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
        
        else:
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
        
        if self.stage == "stage3":
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
                    if self.true_global_step + self.refine_start_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)


    def validation_step(self, batch, batch_idx):
        out = self(batch, phase='val')
        if self.stage == "stage1":
            self.save_image(f"it{self.true_global_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])
        else:
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-{batch['index'][0]}_rgb.png", out["comp_rgb"][0])

    def on_validation_epoch_end(self):
        pass


    # test the gaussians
    def test_step(self, batch, batch_idx):
        if self.stage == "stage1":
            pass
        else:
            pass
            bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]
            background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 新创建的tensor需指定device
            out = self(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])
        

    # save something
    def on_test_epoch_end(self):
        if self.stage == "stage1":
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"it{self.true_global_step}.ply"))
        else:
            self.save_img_sequence(
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step + self.refine_start_step,
            )
            save_path = self.get_save_path(f"it{self.true_global_step + self.refine_start_step}-test/last.ply")
            self.gaussian.save_ply(save_path)

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
            

    def configure_optimizers(self):
        if self.stage == "stage1":
            opt = OptimizationParams(self.parser)
            point_cloud = self.pcd()
            self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
            self.gaussian.training_setup(opt)
            ret = {"optimizer": self.gaussian.optimizer}
        else:
            # load 3dgs from stage 1
            opt = OptimizationParams(self.parser)
            self.gaussian.load_ply(os.path.join(self.log_path, self.cur_time, f"it{self.refine_start_step}.ply"))
            self.gaussian.training_setup(opt)
            ret = {"optimizer": self.gaussian.optimizer}
        return ret
