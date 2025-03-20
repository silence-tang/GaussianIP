from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .models.ip_adapter import IPAdapterPlus
from .models.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from .models.pipeline_ipa import StableDiffusionPipeline
from .models.pipeline_ipa_controlnet import StableDiffusionControlNetPipeline

from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from PIL import Image
import cv2
import os
import numpy as np
import random
from scipy.optimize import minimize

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# seed for generator
seed = 2024

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@threestudio.register("ipa-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_sd_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        pretrained_realistic_model_name_or_path: str = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_path: str = "stabilityai/sd-vae-ft-mse"
        image_encoder_path: str = "IP-Adapter/models/image_encoder"
        image_encoder_faceid_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ip_ckpt_path: str = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
        ip_ckpt_faceid_v1_path: str = "IP-Adapter/models/ip-adapter-faceid_sd15.bin"
        ip_ckpt_faceid_v2_path: str = "IP-Adapter/models/ip-adapter-faceid-plusv2_sd15.bin"

        use_ipa_faceid: bool = True
        use_pose_controlnet: bool = True
        pose_controlnet_path: str = "lllyasviel/control_v11p_sd15_openpose"

        prompt: str = "A girl with black hair."
        negative_prompt: str = "cloned face, multi face, bad face, poorly drawn face, duplicate face, cropped, out of frame, extra fingers, deformed, blurry, bad proportions, disfigured, fused fingers, long neck"
        negative_prompt_faceid: str = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
        null_prompt: str = ""

        pil_image_path: str = "assets/images/girl_face.png"
        pil_image_faceid_path: str = "assets/images/girl.png"
        irr_pil_image_path: str = "assets/images/irr_woman.png"

        batch_size: int = 4
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        ipa_scale: float = 0.6
        ipa_faceid_scale: float = 0.6
        ipa_faceid_s_scale: float = 0.5
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True

        # for sds loss
        use_anpg: bool = False
        weighting_strategy: str = "sds"
    
        # used in prompt_utils.get_text_embeddings
        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4
        lw_depth: float = 0.5
        guidance_rescale: float = 0.0
        original_size: int = 1024
        target_size: int = 1024
        grad_clip_pixel: bool = False
        grad_clip_threshold: float = 0.1

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading IP-Adapter Model ...")

        self.weights_dtype = (torch.float16 if self.cfg.half_precision_weights else torch.float32)
        self.base_model_path = self.cfg.pretrained_realistic_model_name_or_path if self.cfg.use_ipa_faceid else self.cfg.pretrained_sd_model_name_or_path
        self.negative_prompt = self.cfg.negative_prompt_faceid if self.cfg.use_ipa_faceid else self.cfg.negative_prompt
        self.pil_image_path = self.cfg.pil_image_faceid_path if self.cfg.use_ipa_faceid else self.cfg.pil_image_path
        self.irr_pil_image_path = self.cfg.irr_pil_image_path
        self.generator = get_generator(seed, self.device)

        vae = AutoencoderKL.from_pretrained(self.cfg.vae_path).to(self.weights_dtype)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        if self.cfg.use_pose_controlnet:
            pose_controlnet = ControlNetModel.from_pretrained(
                self.cfg.pose_controlnet_path, 
                torch_dtype=self.weights_dtype
            )

            threestudio.info(f"OpenPose ControlNet Model Loaded !")

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model_path,
                controlnet=pose_controlnet,
                torch_dtype=self.weights_dtype,
                scheduler=self.scheduler,
                vae=vae,
                feature_extractor=None,
                safety_checker=None
            ).to(torch_dtype=self.weights_dtype)

            threestudio.info(f"StableDiffusionControlNetPipeline Loaded !")

        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=self.weights_dtype,
                scheduler=self.scheduler,
                vae=vae,
                feature_extractor=None,
                safety_checker=None,
            ).to(torch_dtype=self.weights_dtype)

        # enable memory_efficient_attention
        self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.use_ipa_faceid:
            self.ipa_model = IPAdapterFaceIDPlus(
                self.pipe,
                self.cfg.image_encoder_faceid_path,
                self.cfg.ip_ckpt_faceid_v2_path,
                self.device
            )
        else:
            self.ipa_model = IPAdapterPlus(
                self.pipe,
                self.cfg.image_encoder_path,
                self.cfg.ip_ckpt_path,
                self.device,
                num_tokens=16
            )

        self.ipa_model.set_scale(scale=self.cfg.ipa_faceid_scale if self.cfg.use_ipa_faceid else self.cfg.ipa_scale)

        # 预计算chosen_t (AHDS)
        self.ahds_N = 2400
        self.ahds_t0 = 799
        self.max_t = 800         
        self.tgt_prob_sums = [0.41, 0.21, 0.375]
        self.ranges = [(0, 350), (350, 450), (450, 800)]
        self.init_values = [260, 60, 280],
        self.bounds = [(200, 400), (20, 100), (100, 300)]
        self.optimized_W_p_t_all = self.get_optimized_dual_gaussian(self.init_values, self.tgt_prob_sums, self.ranges, self.max_t, self.bounds)
        self.ahds_chosen_t_all = self.t_scheduler_with_dual_gaussian_pdf(self.optimized_W_p_t_all, N=self.ahds_N, t0=self.ahds_t0)
        self.ahds_chosen_t_min = next((t for t in reversed(self.ahds_chosen_t_all) if t != 0), None)

        cleanup()

        # fetch self.vae and self.unet
        self.vae = self.ipa_model.pipe.vae.eval()
        self.unet = self.ipa_model.pipe.unet.eval()
        
        # we do not need to update the sd model
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.use_pose_controlnet:
            self.controlnet = self.ipa_model.pipe.controlnet.eval()
            for p in self.controlnet.parameters():
                p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        threestudio.info(f"IP-Adapter Model Loaded !")

    
    def prepare_for_sds(self, prompt, negative_prompt, null_prompt):
        if self.cfg.use_ipa_faceid:
            # detect face and get face_embeds (for ipa_faceid)
            self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.pos_image = cv2.imread(self.pil_image_path)
            self.irr_image = cv2.imread(self.irr_pil_image_path)
            self.pos_faces = self.app.get(self.pos_image)
            self.irr_faces = self.app.get(self.irr_image)
            self.pos_faceid_embeds = torch.from_numpy(self.pos_faces[0].normed_embedding).unsqueeze(0)
            self.irr_faceid_embeds = torch.from_numpy(self.irr_faces[0].normed_embedding).unsqueeze(0)
            self.pos_face_image = face_align.norm_crop(self.pos_image, landmark=self.pos_faces[0].kps, image_size=224) # you can also segment the face
            self.irr_face_image = face_align.norm_crop(self.irr_image, landmark=self.irr_faces[0].kps, image_size=224) # you can also segment the face
            # get pos/neg/null image embeds
            self.all_image_embeds = self.ipa_model.get_image_embeds_with_null(
                pos_faceid_embeds=self.pos_faceid_embeds,
                irr_faceid_embeds=self.irr_faceid_embeds,
                pos_face_image=self.pos_face_image,
                irr_face_image=self.irr_face_image, 
                s_scale=self.cfg.ipa_faceid_s_scale,
                shortcut=True
            )
            
        else:
            # load id face
            self.pil_image = Image.open(self.pil_image_path)
            self.pil_image.resize((256, 256))
            # get pos/neg/null image embeds
            self.all_image_embeds = self.ipa_model.get_image_embeds_with_null(
                pil_image=self.pil_image,
                clip_image_embeds=None
            )
            
        # split the returned tuple
        self.pos_image_embeds = self.all_image_embeds[0]
        self.null_image_embeds = self.all_image_embeds[1]
        self.neg_image_embeds = self.all_image_embeds[2]

        self.bs_embed, self.seq_len, _ = self.pos_image_embeds.shape
        self.num_samples = self.cfg.batch_size
        
        # repeat the image_embeds -> [4,16,768]
        self.pos_image_embeds = self.pos_image_embeds.repeat(1, self.num_samples, 1)
        self.pos_image_embeds = self.pos_image_embeds.view(self.bs_embed * self.num_samples, self.seq_len, -1)
        self.neg_image_embeds = self.neg_image_embeds.repeat(1, self.num_samples, 1)
        self.neg_image_embeds = self.neg_image_embeds.view(self.bs_embed * self.num_samples, self.seq_len, -1)
        self.null_image_embeds = self.null_image_embeds.repeat(1, self.num_samples, 1)
        self.null_image_embeds = self.null_image_embeds.view(self.bs_embed * self.num_samples, self.seq_len, -1)

        # get text embeds
        with torch.inference_mode():
            # [4,77,768]
            self.pos_text_embeds, self.neg_text_embeds, self.null_text_embeds = self.ipa_model.pipe.encode_prompt_with_null(
                num_images_per_prompt=self.num_samples,
                prompt=prompt,
                negative_prompt=negative_prompt,
                null_prompt=null_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                null_prompt_embeds=None,
                device=self.device,
            )

            # concat text and image embeds -> [4,93,768]
            self.neg_prompt_embeds = torch.cat([self.neg_text_embeds, self.neg_image_embeds], dim=1)
            self.pos_prompt_embeds = torch.cat([self.pos_text_embeds, self.pos_image_embeds], dim=1)
            self.null_prompt_embeds = torch.cat([self.null_text_embeds, self.null_image_embeds], dim=1)

        # [12,93,768]
        self.final_prompt_embeds_npn = torch.cat([self.neg_prompt_embeds, self.pos_prompt_embeds, self.null_prompt_embeds])
        # [8,93,768]
        self.final_prompt_embeds_np= torch.cat([self.neg_prompt_embeds, self.pos_prompt_embeds])


    # forward_unet
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        noisy_latents: Float[Tensor, "..."],
        control_img: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Optional[Dict[str, Any]],
        use_pose_controlnet: Bool,
        timestep_cond: Float[Tensor, "..."],
        cross_attention_kwargs: Float[Tensor, "..."],
        added_cond_kwargs: Float[Tensor, "..."],
        return_dict: Bool,
    ) -> Float[Tensor, "..."]:
        input_dtype = noisy_latents.dtype

        if not use_pose_controlnet:
            return self.unet(
                noisy_latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict
            ).sample.to(input_dtype)
    
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
                controlnet_cond=control_img.to(self.weights_dtype),
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            return self.unet(
                noisy_latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
                timestep_cond=timestep_cond,
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict
            ).sample.to(input_dtype)
        

    def compute_grad_anpg(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        control_img: Float[Tensor, "B C 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        use_pose_controlnet: Bool,
        all_vis_all: Float[Tensor, "B"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        center: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if self.cfg.view_dependent_prompting:
            text_embeds = prompt_utils.get_text_embeddings(elevation, azimuth, center, all_vis_all, camera_distances, self.cfg.view_dependent_prompting)
            pos_text_embeds= text_embeds[:batch_size * 1]
            neg_text_embeds = text_embeds[batch_size * 1 : batch_size * 2]
            null_text_embeds = text_embeds[batch_size * 2 : batch_size * 3]
            # cat text and image prompt embeds
            pos_prompt_embeds = torch.cat([pos_text_embeds, self.pos_image_embeds], dim=1)
            neg_prompt_embeds = torch.cat([neg_text_embeds, self.neg_image_embeds], dim=1)
            null_prompt_embeds = torch.cat([null_text_embeds, self.null_image_embeds], dim=1)
            # cat neg and pos and null embeds
            final_prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds, null_prompt_embeds])
        else:
            final_prompt_embeds = self.final_prompt_embeds_npn
            
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # expand for anpg
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            control_img = torch.cat([control_img] * 3, dim=0)
            
            # forward unet
            noise_pred = self.forward_unet(
                latent_model_input,
                control_img,
                torch.cat([t] * 3),
                encoder_hidden_states=final_prompt_embeds,
                use_pose_controlnet=use_pose_controlnet,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=True,
            )

            noise_pred_neg, noise_pred_text, noise_pred_null = noise_pred.chunk(3)
            delta_c = self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
            mask = (t < 170).int().view(batch_size, 1, 1, 1)
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (delta_c + delta_d)

        if self.cfg.grad_clip_pixel:
            grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            grad = grad_norm.clamp(max=self.cfg.grad_clip_threshold) * grad / grad_norm

        guidance_eval_utils = {
            "neg_guidance_weights": None,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils


    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        control_img: Float[Tensor, "B C 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        use_pose_controlnet: Bool,
        all_vis_all: Float[Tensor, "B"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        center: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if self.cfg.view_dependent_prompting:
            text_embeds = prompt_utils.get_text_embeddings(elevation, azimuth, center, all_vis_all, camera_distances, self.cfg.view_dependent_prompting)
            pos_text_embeds= text_embeds[:batch_size * 1]
            neg_text_embeds = text_embeds[batch_size * 1 : batch_size * 2]

            # cat text and image prompt embeds
            neg_prompt_embeds = torch.cat([neg_text_embeds, self.neg_image_embeds], dim=1)
            pos_prompt_embeds = torch.cat([pos_text_embeds, self.pos_image_embeds], dim=1)
            
            # cat neg and pos embeds
            final_prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds])

        else:
            final_prompt_embeds = self.final_prompt_embeds_np

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            control_img = torch.cat([control_img] * 2, dim=0)

            # forward unet
            noise_pred = self.forward_unet(
                noisy_latents=latent_model_input,
                control_img=control_img,
                t=torch.cat([t] * 2),
                encoder_hidden_states=final_prompt_embeds,
                use_pose_controlnet=use_pose_controlnet,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=True,
            )

        noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_neg + self.cfg.guidance_scale * (noise_pred_pos - noise_pred_neg)
        
        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale=self.cfg.guidance_rescale)

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "neg_guidance_weights": None,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils


    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 H//8 W//8"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        # imgs should be [-1,1]
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)
    

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents: Float[Tensor, "B 4 64 64"], generator) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        # Decode the latent space to image space
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        # Convert image from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        return image.to(input_dtype)


    def calculate_probability_sums(self, W_p_t_all, ranges):
        # Calculate sums over the given ranges in the format [(start1, end1), (start2, end2), ...]
        sums = [np.sum(W_p_t_all[start:end]) for start, end in ranges]
        return sums


    def error_function(self, params, tgt_prob_sums, ranges, max_t):
        T, s1, s2 = params
        W_p_t_all = np.array([
            np.exp(-(t - T)**2 / (2 * s1**2)) if t <= T else np.exp(-(t - T)**2 / (2 * s2**2))
            for t in range(max_t)
        ])
        W_p_t_all /= np.sum(W_p_t_all)    
        cur_prob_sums = self.calculate_probability_sums(W_p_t_all, ranges)
        error_ratios = sum((cur_prob_sum - tgt_prob_sum) ** 2 for cur_prob_sum, tgt_prob_sum in zip(cur_prob_sums, tgt_prob_sums))
        return error_ratios
        

    def get_optimized_dual_gaussian(self, init_values, tgt_prob_sums, ranges, max_t, bounds):
        result = minimize(
            self.error_function,
            init_values,  # Initial values for T, s1, s2
            args=(tgt_prob_sums, ranges, max_t),
            bounds=bounds,
            method='L-BFGS-B',
            options={'disp': True}
        )

        # Extract the newly optimized parameters
        optimized_T, optimized_s1, optimized_s2 = result.x

        # Generate the optimized W_p_t_all using the newly optimized parameters
        W_p_t_all = np.array([
            np.exp(-(t - optimized_T)**2 / (2 * optimized_s1**2)) if t <= optimized_T else np.exp(-(t - optimized_T)**2 / (2 * optimized_s2**2))
            for t in range(max_t)
        ])

        # Convert to pdf
        W_p_t_all /= np.sum(W_p_t_all)  

        return W_p_t_all


    def t_scheduler_with_dual_gaussian_pdf(self, W_p_t_all, N=2400, t0=899):
        def obj_func(t, i, W_p_t_all):
            t = int(max(0, min(len(W_p_t_all) - 1, t)))
            return abs(sum(W_p_t_all[t:]) - i / N)
        
        chosen_ts = []
        
        for i in range(N):
            selected_t = minimize(obj_func, t0, args=(i, W_p_t_all), method='Nelder-Mead')
            selected_t = max(0, int(selected_t.x))
            chosen_ts.append(selected_t)
        
        return chosen_ts


    def __call__(
        self,
        step,
        rgb: Float[Tensor, "B H W C"],
        control_img: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        use_pose_controlnet: Bool,
        all_vis_all: Float[Tensor, "B"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        center: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        **kwargs,
    ):
        
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        control_img = control_img.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
        
        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW_512.to(self.weights_dtype))
        
        cur_t = self.ahds_chosen_t_all[step]

        if step >= 0 and step < 700:
            t = torch.randint(500, 800, [batch_size], dtype=torch.long, device=self.device)
        elif step >= 700 and step < 900:
            t = torch.randint(400, cur_t+50, [batch_size], dtype=torch.long, device=self.device)
        elif step >= 900 and step < 1400:
            t = torch.randint(150, cur_t+50, [batch_size], dtype=torch.long, device=self.device)
        else:
            # smooth tail
            if cur_t != 0:
                t = torch.randint(20, cur_t+50, [batch_size], dtype=torch.long, device=self.device)
            else:
                t = torch.randint(20, self.ahds_chosen_t_min, [batch_size], dtype=torch.long, device=self.device)


        # anpg or plain sds
        if self.cfg.use_anpg:
            grad, _ = self.compute_grad_anpg(latents, control_img, t, prompt_utils, use_pose_controlnet, all_vis_all, elevation, azimuth, center, camera_distances)
        else:
            grad, _ = self.compute_grad_sds(latents, control_img, t, prompt_utils, use_pose_controlnet, all_vis_all, elevation, azimuth, center, camera_distances)

        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()

        # d(loss_sds)/d(latents) = latents - target = latents - (latents - grad) = grad
        # d(loss_sds)/d(theta) = d(loss_sds)/d(z_t) * d(z_t)/d(theta)
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm()
        }

        return guidance_out
    
    
    def refine_rgb(
        self,
        rgb: Float[Tensor, "refine_n_views H W C"],
        control_img:  Float[Tensor, "refine_n_views H W C"],
        prompt
    ):  
        view_idx_all = [24, 8, 16, 0, 20, 28, 4, 12, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        view_name_all = ['front', 'back', 'left', 'right', 'k0', 'k1', 'k2', 'k3', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23']
        prompt_all = {}
        base_prompt = prompt
        negative_prompt = "blurry face, bad face, poorly drawn face, duplicate face, extra fingers, blurry, fused fingers"
        prompt_all['front'] = base_prompt
        prompt_all['back'] = base_prompt + ', back view'
        prompt_all['left'] = base_prompt + ', left view'
        prompt_all['right'] = base_prompt + ', right view'
        prompt_all['k0'] = base_prompt + ', left front view'
        prompt_all['k1'] = base_prompt + ', right front view'
        prompt_all['k2'] = base_prompt + ', right back view'
        prompt_all['k3'] = base_prompt + ', left back view'
        for view_name in view_name_all[8:]:
            prompt_all[view_name] = base_prompt

        key_view_name_pair_mapper = {
            'v0': ('left', 'k0'), 'v1': ('left', 'k0'), 'v2': ('left', 'k0'), 'v3': ('k0', 'front'),
            'v4': ('k0', 'front'), 'v5': ('k0', 'front'), 'v6': ('front', 'k1'), 'v7': ('front', 'k1'),
            'v8': ('front', 'k1'), 'v9': ('k1', 'right'), 'v10': ('k1', 'right'), 'v11': ('k1', 'right'),
            'v12': ('right', 'k2'), 'v13': ('right', 'k2'), 'v14': ('right', 'k2'), 'v15': ('k2', 'back'),
            'v16': ('k2', 'back'), 'v17': ('k2', 'back'), 'v18': ('back', 'k3'), 'v19': ('back', 'k3'),
            'v20': ('back', 'k3'), 'v21': ('k3', 'left'), 'v22': ('k3', 'left'), 'v23': ('k3', 'left')}
        key_view_weight_pair_mapper = {
            'v0': (0.75, 0.25), 'v1': (0.5, 0.5), 'v2': (0.25, 0.75), 'v3': (0.75, 0.25),
            'v4': (0.5, 0.5), 'v5': (0.25, 0.75), 'v6': (0.75, 0.25), 'v7': (0.5, 0.5),
            'v8': (0.25, 0.75), 'v9': (0.75, 0.25), 'v10': (0.5, 0.5), 'v11': (0.25, 0.75),
            'v12': (0.75, 0.25), 'v13': (0.5, 0.5), 'v14': (0.25, 0.75), 'v15': (0.75, 0.25),
            'v16': (0.5, 0.5), 'v17': (0.25, 0.75), 'v18': (0.75, 0.25), 'v19': (0.5, 0.5),
            'v20': (0.25, 0.75), 'v21': (0.75, 0.25), 'v22': (0.5, 0.5), 'v23': (0.25, 0.75)}

        unet = self.ipa_model.pipe.unet

        tgt_attn_processors = [
            unet.attn_processors['up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor'],
            unet.attn_processors['up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor'],
        ]

        num_steps = 8
        lambda_self = 0.55

        for processor in tgt_attn_processors:
            processor.state = 'refine'
            processor.total_denoise_step = num_steps
            processor.lambda_self = lambda_self
            
        ##################################################################################################
        rgb_height = rgb.shape[1]
        rgb_width = rgb.shape[2]

        # get the device to do refine
        refine_device = f"cuda:{get_rank() + 1}"
        
        # reshape
        rgb_BCHW = rgb.permute(0, 3, 1, 2).to(refine_device)            # [refine_n_views, 3, 1024, 1024]
        control_img = control_img.permute(0, 3, 1, 2).to(refine_device) # [refine_n_views, 3, 1024, 1024]

        # Step 1: prepare timesteps
        timesteps = torch.linspace(0, 999, 50, dtype=torch.int64, device=refine_device).round().flip(dims=[0])
        timesteps_sub = timesteps[-num_steps:]
        t = timesteps_sub[0].clone().detach().reshape(1).to(torch.long).to(refine_device)
        num_inference_steps = len(timesteps)

        # Step 2: prepare the same noise for all the images
        noise = torch.randn(1, 4, rgb_height // 8, rgb_width // 8, device=refine_device, dtype=torch.float16)
        noise = noise.repeat(1, 1, 1, 1)

        # Step 3: prepare negative prompt (different from the main one)
        negative_prompt = "blurry face, bad face, poorly drawn face, duplicate face, extra fingers, blurry, fused fingers"
        refined_rgbs = []

        # Step 4: refine images
        with torch.no_grad():
            self.vae.to(refine_device)
            self.ipa_model.device = refine_device
            self.ipa_model.pipe.to(refine_device)
            self.ipa_model.image_encoder.to(refine_device)
            self.ipa_model.image_proj_model.to(refine_device)

            for i, view_info in enumerate(zip(view_idx_all, view_name_all)):
                view_idx = view_info[0]
                view_name = view_info[1]
                print('Refining {}th image, view_idx: {}, view_name: {}'.format(i, view_idx, view_name))

                # encode image into latents with vae
                cur_rgb_BCHW = rgb_BCHW[view_idx].reshape(1, 3, rgb_height, rgb_width)
                cur_control_img = control_img[view_idx].reshape(1, 3, rgb_height, rgb_width)
                cur_latents = self.encode_images(cur_rgb_BCHW.to(self.weights_dtype))
                cur_latents_noisy = self.scheduler.add_noise(cur_latents, noise, t)
                
                for processor in tgt_attn_processors:
                    processor.cur_view_name = view_name
                    processor.stored_zt[view_name] = []
                    if 'v' in view_name:
                        processor.cur_key_view_name_pair = key_view_name_pair_mapper[view_name]
                        processor.cur_key_view_weight_pair = key_view_weight_pair_mapper[view_name]
                
                cur_prompt = prompt_all[view_name]

                refined_rgb = self.ipa_model.refine_with_small_noise(
                    latents=cur_latents_noisy,
                    prompt=cur_prompt,
                    negative_prompt=negative_prompt,
                    face_image=self.pos_face_image,
                    faceid_embeds=self.pos_faceid_embeds,
                    image=cur_control_img,
                    shortcut=True,
                    scale=0.6,
                    s_scale=0.5,
                    num_samples=1,
                    width=1024,
                    height=1024,
                    timesteps=timesteps_sub,
                    num_inference_steps=num_inference_steps,
                    seed=2024
                )
            
                refined_rgbs.append(refined_rgb.to(self.device))

        self.vae.to(self.device)
        self.ipa_model.device = self.device
        self.ipa_model.pipe.to(self.device)
        self.ipa_model.image_encoder.to(self.device)
        self.ipa_model.image_proj_model.to(self.device)

        # concat all the images
        refined_rgbs = torch.cat(refined_rgbs, dim=0)  # [refine_n_views, 3, 1024, 1024]

        torch.cuda.empty_cache()
        cleanup()

        return refined_rgbs.permute(0, 2, 3, 1), view_idx_all