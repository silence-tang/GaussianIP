# GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior

[Zichen Tang](https://github.com/silence-tang)<sup>1</sup>, [Yuan Yao]()<sup>2</sup>, [Miaomiao Cui]()<sup>2</sup>, [Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=en)<sup>2</sup>, [Hongyu Yang](https://scholar.google.com/citations?user=dnbjaWIAAAAJ)<sup>1</sup>.  
<sup>1</sup>Beihang University&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Alibaba Group

### [Project](https://alvinliu0.github.io/projects/HumanGaussian) | [arXiv](https://arxiv.org/abs/2311.17061)

Text-guided 3D human generation has advanced with the development of efficient 3D representations and 2D-lifting methods like Score Distillation Sampling (SDS). However, current methods suffer from prolonged training times and often produce results that lack fine facial and garment details. In this paper, we propose GaussianIP, an effective two-stage framework for generating identity-preserving realistic 3D humans from text and image prompts. Our core insight is to leverage human-centric knowledge to facilitate the generation process. In stage 1, we propose a novel **Adaptive Human Distillation Sampling (AHDS)** method to rapidly generate a 3D human that maintains high identity consistency with the image prompt and achieves a realistic appearance. Compared to traditional SDS methods, AHDS better aligns with the human-centric generation process, enhancing visual quality with notably fewer training steps. To further improve the visual quality of the face and clothes regions, we design a **View-Consistent Refinement (VCR)** strategy in stage 2. Specifically, it produces detail-enhanced results of the multi-view images from stage 1 iteratively, ensuring the 3D texture consistency across views via mutual attention and distance-guided attention fusion. Then a polished version of the 3D human can be achieved by directly perform reconstruction with the refined images. **Extensive experiments demonstrate that GaussianIP outperforms existing methods in both visual quality and training efficiency, particularly in generating identity-preserving results.**

## Installation
```
# clone the github repo
git clone https://github.com/silence-tang/GaussianIP.git
cd GaussianIP

# install torch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt

# install a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

## Text Prompts Gallery
The text prompts that are used for qualitative/ablation visual results demonstration are included in `gallery_text_prompts.txt`.

## Image Prompts Gallery
The image prompts that are used for qualitative/ablation visual results demonstration are included in `gallery_image_prompts.txt`.

## Pretrained Models
Please prepare below pre-trained models before the training process:

* **SMPL-X**: Download SMPL-X model from https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip, unzip and place smpl under `/path/to/GaussianIP`.

* **stabilityai/sd-vae-ft-mse**: Download VAE model from https://huggingface.co/stabilityai/sd-vae-ft-mse.

* **SG161222/Realistic_Vision_V4.0_noVAE**: Download diffusion base model from https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE.

* **lllyasviel/control_v11p_sd15_openpose**: Download ControlNet model from https://huggingface.co/lllyasviel/control_v11p_sd15_openpose.

* **laion/CLIP-ViT-H-14-laion2B-s32B-b79K**: Download image encoder model from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K.

* **ip-adapter-faceid-plusv2_sd15.bin**: Download ip-adapter model from https://huggingface.co/h94/IP-Adapter-FaceID.

After the above models are downloaded, please specify their paths in `configs/exp.yaml` and `threestudio/models/guidance/refine.py` properly.

## Train
First, specify `TEXT_PROMPT` and `IMAGE_PROMPT` in run.sh.
Then run:
```bash
bash run.sh
```

## Animation
### Animation-Related Installation
```bash
# for GUI
pip install dearpygui

# cubvh
pip install git+https://github.com/ashawkey/cubvh

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install -U git+https://github.com/ashawkey/kiuikit

# smplx
pip install smplx[all]
# please also download SMPL-X files to ./smplx_models/smplx/*.pkl
```

### Animation-Related Usage
Motions should follow the SMPL-X body pose format (21 joints), which could read by:
```python
motion = np.load('content/amass_test_17.npz')['poses'][:, 1:22, :3]
# motion = np.load('content/Aeroplane_FW_part9.npz')['poses'][:, 1:22, :3]
```

Then, perform the zero-shot animating by:
```bash
# visualize in gui
python animation.py --ply <path/to/ply> --motion <path/to/motion> --gui

# play motion and save to videos/xxx.mp4
python animation.py --ply <path/to/ply> --motion <path/to/motion> --play

# also self-rotate during playing
python animation.py --ply <path/to/ply> --motion <path/to/motion> --play --rotate
```

## Acknowledgement
This work is inspired by and builds upon numerous outstanding research efforts and open-source projects, including [Threestudio](https://github.com/threestudio-project/threestudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), [HumanGaussian](https://github.com/alvinliu0/HumanGaussian/), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/). We are deeply grateful to all the authors for their contributions and for sharing their work openly!

## Notes
We train on the resolution of 1024x1024 with a batch size of 4. The whole optimization process takes around 40 minutes on a single NVIDIA V100 (32GB) GPU or a single NVIDIA RTX 3090 (24GB) GPU.

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a star.
```
@misc{
    tang2025gaussianipidentitypreservingrealistic3d,
    title={GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior}, 
    author={Zichen Tang and Yuan Yao and Miaomiao Cui and Liefeng Bo and Hongyu Yang},
    year={2025},
    eprint={2503.11143},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.11143}, 
}
