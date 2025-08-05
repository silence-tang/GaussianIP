#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from gs_renderer import GaussianModel
import torch
from custom_smplx.smplx_custom import SMPLX
import numpy as np
from utils.general import create_video

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
# from body_visualizer.tools.vis_tools import show_image

# import smplx

from utils.rotations import (
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_axis_angle,
)
from avatar.utils import (
    get_rotating_camera,
)
from tqdm import tqdm
from utils.geometry import transformations
import torchvision
from gaussiansplatting.gaussian_renderer import render, render_deformed
from gaussiansplatting.scene.cameras import MiniCam
from pytorch3d.ops.knn import knn_points
from arguments import ModelParams
SCALE_Z = 1e-5

SMPL_PATH = 'data/smpl'
SMPLX_PATH = '/root/autodl-tmp/smplx'

# NOTE: 需要调成smplx的
AMASS_SMPLH_TO_SMPL_JOINTS = np.arange(0, 156).reshape((-1, 3))[[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,   22, 37
]].reshape(-1)

class Avatar:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel):
        """
        :param path: Path to colmap scene main folder.
        """
        self.device = 'cuda'
        self.gaussians = gaussians

        # 加载smpl模型
        # self.smpl = SMPL(SMPL_PATH).to(self.device)
        # self.smplx = smplx.SMPLX(SMPLX_PATH).to(self.device)
        self.smplx = SMPLX(
            SMPLX_PATH, 
            gender='neutral', 
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='npz',
            flat_hand_mean=True,
        ).to(self.device)
        self.ori_center = None

        self.get_A_verts()
        # self.get_vitruvian_verts()

        self.vposer, self.ps = load_model('/root/autodl-tmp/GaussianIP/utils/human_body_prior/V02_05', model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        self.vposer = self.vposer.to('cuda')

    @torch.no_grad()
    def get_vitruvian_verts(self):
        vitruvian_pose = get_predefined_pose('t_pose', self.device)[0]
        vitruvian_pose = torch.zeros((1, 69), dtype=torch.float32, device='cuda')
        vitruvian_pose[:, 2] = 0.05
        vitruvian_pose[:, 5] = -0.05
        vitruvian_pose[:, 47] = -0.4
        vitruvian_pose[:, 50] = 0.4
        vitruvian_pose[:, 48] = -0.78
        vitruvian_pose[:, 51] = -0.78

        betas = [-0.5773,  0.1266, -0.5128,  0.1527,  2.3070, -2.3127, -0.6074, -1.1296,
         -0.8285,  0.1158]
        betas = torch.zeros(10)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)

        smpl_output = self.smplx(body_pose=vitruvian_pose, betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)

        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0]
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()
    

    @torch.no_grad()
    def get_A_verts(self):
        a_body_pose = np.zeros((21, 3), dtype=np.float32)
        a_body_pose[0, 1] = 0.2
        a_body_pose[0, 2] = 0.1
        a_body_pose[1, 1] = -0.2
        a_body_pose[1, 2] = -0.1
        a_body_pose[15, 2] = -0.7853982
        a_body_pose[16, 2] = 0.7853982
        # a_body_pose[19, 0] = 1.0
        # a_body_pose[20, 0] = 1.0

        # betas = None
        # expression=None

        betas = torch.zeros([1, 10])
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        expression = torch.zeros([1, 10])
        self.expression = torch.tensor(expression, dtype=torch.float32, device=self.device)

        smplx_output = self.smplx(
            body_pose=torch.tensor(a_body_pose, dtype=torch.float32, device=self.device).unsqueeze(0),
            betas=self.betas,
            expression=self.expression,
            return_verts=True
        )

        a_verts = smplx_output.vertices[0]
        a_verts_numpy = a_verts.detach().cpu().numpy()

        # rescale and recenter
        if self.ori_center is None:
            vmin = a_verts_numpy.min(0)
            vmax = a_verts_numpy.max(0)
            self.ori_center = torch.tensor((vmax + vmin) / 2, device='cuda') # [-9.793043e-05, -0.43814677, -0.005535446]
            self.ori_scale = 0.6 / np.max(vmax - vmin) # 0.3500760954126092

        # NOTE: 先把高斯变换到smplx位置？
        self.unscale_points(self.gaussians._xyz, -10)
        self.gaussians._xyz = self.gaussians._xyz / self.ori_scale + self.ori_center
        
        self.T_t2a = smplx_output.T[0].detach()
        self.inv_T_t2a = torch.inverse(self.T_t2a)
        
        self.canonical_offsets = smplx_output.shape_offsets + smplx_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0]
        self.a_verts = a_verts.detach()
        # self.a_verts = (self.a_verts - self.ori_center) * self.ori_scale
        # self.scale_verts(-10)
        return a_verts.detach()

    def scale_verts(self, delta):
        # self.points3D[:, :3] *= 1.1 ** (-delta)
        if self.a_verts is not None:
            self.a_verts *= 1.1 ** (-delta)
    
    def unscale_points(self, points, delta):
        # self.points3D[:, :3] *= 1.1 ** (-delta)
        if points is not None:
            points /= 1.1 ** (-delta)

    @torch.no_grad()
    def render_canonical(self, nframes=180, pose_type=None, pipe=None, bg=None):
        iter_s = 'final'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        os.makedirs('./output/canon/', exist_ok=True)

        # 获取渲染canonical人物时旋转的相机参数, 每一帧都有一个相机参数
        # NOTE: 需要和gaussianip使用的相机参数对齐
        # NOTE: 还需探究fovx, fovy等参数对渲染结果的影响
        camera_params = get_rotating_camera(
            img_size=1024,
            fov=0.87266, # animation.py是50 deg, 训练是70 deg
            # fov=1.222, # 70 deg
            dist=2.5,
            # znear=0.1,  # 训练
            # zfar=1000.0, # 训练
            znear=0.01, # animation.py
            zfar=100.0, # animation.py
            device='cuda',
            nframes=nframes,
            angle_limit=2 * torch.pi,
        )

        pbar = tqdm(range(nframes), desc="Canonical:")
        for idx in pbar:
            cam_p = camera_params[idx]
            fovx = cam_p["fovx"]
            fovy = cam_p["fovy"]
            width = cam_p["image_width"]
            height = cam_p["image_height"]
            world_view_transform = cam_p["world_view_transform"]
            full_proj_transform = cam_p["full_proj_transform"]

            # 将每帧的相机参数封装为MiniCam对象(viewpoint_cam)

            # world_view_transform(animation.py):
            # tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
            #         [ 0.0000, -1.0000, -0.0000,  0.0000],
            #         [ 0.0000, -0.0000, -1.0000,  0.0000],
            #         [-0.0000, -0.0000,  2.5000,  1.0000]], device='cuda:0')
            
            # full_proj_transform(animation.py):
            # tensor([[ 2.1445,  0.0000,  0.0000,  0.0000],
            #         [ 0.0000, -2.1445,  0.0000,  0.0000],
            #         [ 0.0000,  0.0000, -1.0001, -1.0000],
            #         [ 0.0000,  0.0000,  2.4902,  2.5000]], device='cuda:0')

            # world_view_transform(__init__.py):
            # tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
            #         [ 0.0000, -1.0000,  0.0000,  0.0000],
            #         [ 0.0000,  0.0000, -1.0000,  0.0000],
            #         [ 0.0000,  0.0000,  2.5000,  1.0000]], device='cuda:0')
            
            # full_proj_transform(__init__.py):
            # tensor([[ 2.1445,  0.0000,  0.0000,  0.0000],
            #         [ 0.0000, -2.1445,  0.0000,  0.0000],
            #         [ 0.0000,  0.0000, -1.0001, -1.0000],
            #         [ 0.0000,  0.0000,  2.4902,  2.5000]], device='cuda:0')

            viewpoint_cam = MiniCam(width, height, fovy, fovx, 0.1, 100.0, world_view_transform, full_proj_transform)
            # 把加载的高斯原封不动丢进去执行渲染
            # TODO: 改成在smplx表面初始化高斯然后渲染, 并查看初始化的高斯与smplx表面的最近邻vs加载的高斯与smplx表面的最近邻
            # TODO: 将smplx顶点进行与animation.py中同样的变换, 并查看加载的高斯与smplx表面的最近邻距离是否小于smplx顶点不变换的距离
            render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg)
            image = render_pkg["render"]
            # 存图
            torchvision.utils.save_image(image, f'./output/canon/{idx:05d}.png')

        video_fname = f'./output/canon/canon_{iter_s}.mp4'
        create_video(f'./output/canon/', video_fname, fps=30)


    @torch.no_grad()
    def animate(self, motion_path=None, pipe=None, bg=None):
        iter_s = 'final'
        os.makedirs('./output/anim/', exist_ok=True)

        # NOTE: AMASS的SFU(SMPL-X G)应该能与smplx兼容, 查看SMPL-H和SMPL-X序列的区别
        motions = np.load(motion_path)
        motion_len = motions['root_orient'].shape[0]
        start_idx = 0
        end_idx = motion_len // 4
        skip = 4

        # root_orient
        root_orient = torch.tensor(motions['root_orient'][start_idx:end_idx:skip], dtype=torch.float32, device='cuda')
        ro_value = torch.tensor([0, 0, 0], device='cuda')
        root_orient = ro_value.repeat(root_orient.shape[0], 1)

        # transl
        transl = torch.tensor(motions['trans'][start_idx:end_idx:skip], dtype=torch.float32, device='cuda')
        transl_value = torch.tensor([0.0, 0.0, 0.0], device='cuda')
        transl = transl_value.repeat(root_orient.shape[0], 1)

        # betas
        betas = torch.zeros([root_orient.shape[0], 3], dtype=torch.float32, device='cuda')

        # pose
        pose_body = torch.tensor(motions['pose_body'][start_idx:end_idx:skip], dtype=torch.float32, device='cuda')

        # body_pose = torch.zeros((21, 3), dtype=torch.float32, device='cuda')
        # body_pose[0, 1] = 0.2
        # body_pose[0, 2] = 0.1
        # body_pose[1, 1] = -0.2
        # body_pose[1, 2] = -0.1
        # body_pose[15, 2] = -0.7853982
        # body_pose[16, 2] = 0.7853982
        # body_pose[19, 0] = 1.0
        # body_pose[20, 0] = 1.0
        # body_pose = body_pose.reshape(63)
        # pose_body = body_pose.repeat(root_orient.shape[0], 1)

        # load body model
        # bm = BodyModel(bm_fname='/root/autodl-tmp/smplx/SMPLX_NEUTRAL.npz').to('cuda')

        # # 随机采样pose
        # pose_body_all = []
        # for i in range(pose_body.shape[0]):
        #     latent_pose_code = torch.from_numpy(np.random.randn(1, 32).astype(np.float32)).to('cuda')
        #     pose_body = self.vposer.decode(latent_pose_code)['pose_body'].contiguous().view(-1, 63)
        #     # images = render_smpl_params(bm, {'pose_body':pose_body}).reshape(1,1,1,400,400,3)
        #     # imagearray2file(images, f'/root/autodl-tmp/GaussianIP/output/anim/test_pose_body_{i}.png')
        #     pose_body_all.append(pose_body.reshape(63))
        #     # 将pose写入text中, 方便根据图找到索引再找到对应的pose参数
        
        # with open('/root/autodl-tmp/GaussianIP/human_body_prior/tutorials/output/anim/pose.txt', 'w') as file:
        #     for pose_body in pose_body_all:
        #         file.write(f"{pose_body.tolist()}\n")

        # pose_body = torch.stack(pose_body_all, dim=0).to('cuda')

        # NOTE: 定义渲染视频所需的相机参数, 需确认是否与gaussianip对齐
        camera_params = get_rotating_camera(
            img_size=1024,
            fov=0.87266, # animation.py是50 deg, 训练是70 deg
            # fov=1.222,
            dist=2.5,
            # znear=0.1,  # 训练
            # zfar=1000.0, # 训练
            znear=0.01, # animation.py
            zfar=100.0, # animation.py
            device='cuda',
            nframes= len(root_orient),
            angle_limit=2 * torch.pi,
        )

        ext_tfs = None
        # manual_trans = torch.tensor((np.array([0, -1, 0])), dtype=torch.float32, device='cuda') # 不然人会飘在天上
        # manual_rot = np.array([-90.0, 0, 0]) / 180 * np.pi
        # manual_rotmat = torch.tensor((transformations.euler_matrix(*manual_rot)[:3, :3]),dtype=torch.float32, device='cuda')
        # manual_scale = torch.tensor((1.0), dtype=torch.float32, device='cuda')
        # ext_tfs = (manual_trans.unsqueeze(0), manual_rotmat.unsqueeze(0), manual_scale.unsqueeze(0))
        
        for idx in tqdm(range(pose_body.shape[0]), desc="Animation"):
            print(idx)

            # 对人体高斯执行lbs变换, 传入当前帧的global_orient, body_pose, betas, expression等参数
            # NOTE: 先忽略手部动作
            output = self.forward(
                global_orient=root_orient[idx-1, :],
                body_pose=pose_body[idx-1, :],
                left_hand_pose=None,
                right_hand_pose=None,
                jaw_pose=None,
                leye_pose=None,
                reye_pose=None,
                betas=self.betas,
                expression=self.expression,
                transl=transl[idx-1],
                # NOTE: 如果设置为None, 则人物位置偏下
                # transl=None, 
                ext_tfs=ext_tfs,
            )

            feats = output['shs']
            means3D = output['xyz']
            opacity = output['opacity']
            scales = output['scales']
            rotations = output['rotq']
            active_sh_degree = output['active_sh_degree']

            # # debug: 用cano属性去渲染
            # feats = output['shs']
            # means3D = output['xyz_canon']
            # opacity = output['opacity']
            # scales = output['scales_canon']
            # rotations = output['rotq_canon']
            # active_sh_degree = output['active_sh_degree']

            # NOTE: 每帧都用相同的相机参数进行渲染(azi=0, 正面视角)
            # cam_p = camera_params[idx]
            cam_p = camera_params[0]

            # 解析出当前使用的相机参数
            fovx = cam_p["fovx"]           # 0.873
            fovy = cam_p["fovy"]           # 0.873
            width = cam_p["image_width"]   # 1024
            height = cam_p["image_height"] # 1024
            world_view_transform = cam_p["world_view_transform"]
            full_proj_transform = cam_p["full_proj_transform"]

            viewpoint_cam = MiniCam(width, height, fovy, fovx, 0.1, 100.0, world_view_transform, full_proj_transform)
            
            # 执行渲染
            render_pkg = render_deformed(viewpoint_cam, feats, means3D, opacity, scales, rotations, active_sh_degree, pipe, bg)
            image = render_pkg["render"]
            # 存图
            torchvision.utils.save_image(image, f'./output/anim/{idx:05d}.png')

        video_fname = f'./output/anim/anim_{iter_s}.mp4'
        create_video(f'./output/anim/', video_fname, fps=20)

    def forward(
        self,
        global_orient=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        betas=None,
        expression=None,
        transl=None,
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        gs_scales = self.gaussians.scaling_activation(self.gaussians._scaling)
        gs_rotq = self.gaussians.rotation_activation(self.gaussians._rotation)
        gs_xyz = self.gaussians._xyz
        gs_opacity = self.gaussians.opacity_activation(self.gaussians._opacity)
        gs_shs = self.gaussians.get_features

        gs_scales_canon = gs_scales.clone()
        gs_rotmat = quaternion_to_matrix(gs_rotq)

        # if hasattr(self, 'global_orient') and global_orient is None:
        #     global_orient = rotation_6d_to_axis_angle(self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)

        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23 * 3)

        if hasattr(self, 'betas') and betas is None:
            betas = self.betas

        if hasattr(self, 'expression') and expression is None:
            expression = self.expression

        # if hasattr(self, 'transl') and transl is None:
        #     transl = self.transl[dataset_idx]

        # a-pose -> t-pose -> tgt-pose
        # remove & reapply the blendshape

        # NOTE: 表情参数暂时固定为self.expression(全0), 看动作数据集是否提供表情参数
        # NOTE: jaw/leye/reye先置None
        smplx_output = self.smplx(
            global_orient=global_orient.unsqueeze(0),
            betas=self.betas,
            body_pose=body_pose.unsqueeze(0),
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=self.expression,
            return_verts=True,
            disable_posedirs=False,
        )

        curr_offsets = (smplx_output.shape_offsets + smplx_output.pose_offsets)[0]
        T_t2pose = smplx_output.T[0]
        T_a2t = self.inv_T_t2a.clone()
        T_a2t[..., :3, 3] = T_a2t[..., :3, 3] + self.canonical_offsets - curr_offsets
        T_a2pose = T_t2pose @ T_a2t # NOTE: 注意是标准姿态下的a2pose变换，而不是recenter&rescale后的a2pose变换
        
        # T_t2pose = smplx_output.T[0]
        # T_vitruvian2t = self.inv_T_t2vitruvian.clone()
        # T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
        # T_vitruvian2pose = T_t2pose @ T_vitruvian2t

        # TODO: 改成在smplx表面初始化高斯然后渲染, 并查看初始化的高斯与smplx表面的最近邻vs加载的高斯与smplx表面的最近邻
        # TODO: 将smplx顶点进行与animation.py中同样的变换, 并查看加载的高斯与smplx表面的最近邻距离是否小于smplx顶点不变换的距离
        lbs_gau = smplx_lbs_diffuse_gau_topk(
            lbs_weights=self.smplx.lbs_weights,         # 固定的smplx顶点的lbs权重
            verts_transform=T_a2pose.unsqueeze(0),      # 所有smplx顶点的a2pose变换
            points=gs_xyz.unsqueeze(0),                 # 标准姿态下的高斯位置
            template_points=self.a_verts.unsqueeze(0),  # 标准姿态下的mesh顶点位置
            K=6
        )
        
        # lbs_gau = smplx_lbs_diffuse_gau_topk(
        #     lbs_weights=self.smplx.lbs_weights,         # 固定的smplx顶点的lbs权重
        #     verts_transform=T_vitruvian2pose.unsqueeze(0),      # 所有smplx顶点的a2pose变换
        #     points=gs_xyz.unsqueeze(0),                 # 标准姿态下的高斯位置
        #     template_points=self.vitruvian_verts.unsqueeze(0),  # 标准姿态下的mesh顶点位置
        #     K=6
        # )

        lbs_gau = lbs_gau.squeeze(0)

        homogen_coord = torch.ones_like(gs_xyz[..., :1])
        gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
        deformed_xyz = torch.matmul(lbs_gau, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)

        deformed_gs_rotmat = lbs_gau[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        if ext_tfs is not None:
            # NOTE: 用外部自定义的后处理方式对xyz, rot, scale进行最终处理
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales

            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        self.gaussians.normals = torch.zeros_like(gs_xyz)
        self.gaussians.normals[:, 2] = 1.0

        deformed_gs_shs = gs_shs

        return {
            'xyz': deformed_xyz,  # deform之后的高斯位置
            'xyz_canon': gs_xyz,  # 标准姿态下的高斯位置
            'xyz_offsets': torch.zeros_like(gs_xyz),
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'active_sh_degree': self.gaussians.active_sh_degree,
        }

def get_predefined_pose(pose_type, device):
    if pose_type == 'da_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)
        body_pose[:, 2] = 1.0
        body_pose[:, 5] = -1.0
    elif pose_type == 'a_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)
        body_pose[:, 2] = 0.2
        body_pose[:, 5] = -0.2
        body_pose[:, 47] = -0.8
        body_pose[:, 50] = 0.8
    elif pose_type == 't_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)

    return body_pose

def get_predefined_smplx_pose(pose_type, device):
    if pose_type == 'a_pose':
        body_pose = torch.zeros((21, 3), dtype=torch.float32, device=device)
        body_pose[0, 1] = 0.2
        body_pose[0, 2] = 0.1
        body_pose[1, 1] = -0.2
        body_pose[1, 2] = -0.1
        body_pose[15, 2] = -0.7853982
        body_pose[16, 2] = 0.7853982
        body_pose[19, 0] = 1.0
        body_pose[20, 0] = 1.0
    elif pose_type == 't_pose':
        body_pose = torch.zeros((21, 3), dtype=torch.float32, device=device)
    return body_pose

def batch_index_select(data, inds):
    bs, nv = data.shape[:2]
    device = data.device
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]


# def smplx_lbs_diffuse_gau_topk(
#     lbs_weights,
#     verts_transform,
#     points,
#     template_points,
#     K=6,
#     addition_info=None
# ):
#     '''ref: https://github.com/JanaldoChen/Anim-NeRF
#     Args:
#     '''
#     bz, np, _ = points.shape
#     with torch.no_grad():
#         results = knn_points(points, template_points, K=K)
#         dists, idxs = results.dists, results.idx

#     neighbs_dist = dists
#     neighbs = idxs
#     weight_std = 0.1
#     weight_std2 = 2. * weight_std ** 2
#     xyz_neighbs_lbs_weight = lbs_weights[neighbs]  # (bs, n_rays*K, k_neigh, 24)
#     xyz_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1) / weight_std2)  # (bs, n_rays*K, k_neigh)
#     xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
#     xyz_neighbs_weight = torch.exp(-neighbs_dist)  # (bs, n_rays*K, k_neigh)
#     xyz_neighbs_weight *= xyz_neighbs_weight_conf
#     xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True)  # (bs, n_rays*K, k_neigh)

#     xyz_neighbs_transform = batch_index_select(verts_transform, neighbs)  # (bs, n_rays*K, k_neigh, 4, 4)
#     xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform, dim=2)  # (bs, n_rays*K, 4, 4)
#     xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True)  # (bs, n_rays*K, 1)

#     if addition_info is not None:  # [bz, nv, 3]
#         xyz_neighbs_info = batch_index_select(addition_info, neighbs)
#         xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1) * xyz_neighbs_info, dim=2)
#         return xyz_dist, xyz_transform, xyz_info
#     else:
#         return xyz_dist, xyz_transform
    
def smplx_lbs_diffuse_gau_topk(
    lbs_weights,
    verts_transform,
    points,
    template_points,
    K=6,
    addition_info=None
):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    '''
    bz, np, _ = points.shape

    # points[0].cpu().numpy().min(0)
    # array([-0.5872435 , -0.8470391 , -0.16731435], dtype=float32)
    # points[0].cpu().numpy().max(0)
    # array([0.59194416, 0.82413304, 0.17705269], dtype=float32)

    # template_points.cpu().numpy()[0].min(0)
    # array([-0.5637325 , -0.7781227 , -0.13287593], dtype=float32)
    # template_points.cpu().numpy()[0].max(0)
    # array([0.5637325 , 0.7781227 , 0.13287593], dtype=float32)

    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx

    NNs_dist = dists
    NNs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    conf_threshold = 0.5

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
    gau_NNs_transform = batch_index_select(verts_transform, NNs)  # [1, gau_num, k, 4, 4]
    # 计算这些变换矩阵的加权和, [1, gau_num, 4, 4]
    gau_transform = torch.sum(gau_NNs_weights.unsqueeze(-1).unsqueeze(-1) * gau_NNs_transform, dim=2)
    
    return gau_transform
