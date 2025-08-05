import torch
import sys
from avatar import Avatar
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gs_renderer import Renderer, MiniCam

def animate(dataset, pipe):
    # gaussian model
    renderer = Renderer(sh_degree=0, white_background=False)
    renderer.gaussians.load_ply_ori('/root/autodl-tmp/GaussianIP/data/humans/last.ply')

    scene = Avatar(dataset, renderer.gaussians)
    # NOTE: 这边似乎用load_ply_new会导致飘出来的身体部分
    # NOTE: 原始3dgs, humangaussian anime, anime-gs都用的这个

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # motion_path = '/root/autodl-tmp/GaussianIP/data/SFU_SMPLX/0007/0007_Crawling001_stageii.npz'
    # motion_path = '/root/autodl-tmp/GaussianIP/data/SFU_SMPLX/0007/0007_Cartwheel001_stageii.npz'
    # motion_path = '/root/autodl-tmp/GaussianIP/data/SFU_SMPLX/0005/0005_SideSkip001_stageii.npz'
    motion_path = '/root/autodl-tmp/GaussianIP/data/SFU_SMPLX/0008/0008_Yoga001_stageii.npz'
    
    scene.animate(motion_path, pipe=pipe, bg=background)
    # scene.render_canonical(nframes=180, pose_type=None, pipe=pipe, bg=background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="rendering script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])
    args.test = True
    print("Optimizing " + args.model_path)

    animate(lp.extract(args), pp.extract(args))

    # All done
    print("\nrender complete.")
