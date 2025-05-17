import torch
import sys
from avatar import Avatar
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gs_renderer import Renderer, MiniCam

def animate(dataset, pipe):
    # gaussian model
    renderer = Renderer(sh_degree=0, white_background=False)

    scene = Avatar(dataset, renderer.gaussians)
    scene.gaussians.load_ply_ori('/root/autodl-tmp/GaussianIP/data/humans/last.ply')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    motion_path = '/root/autodl-tmp/GaussianIP/data/SFU_SMPLX/0007/0007_Crawling001_stageii.npz'
    scene.animate(motion_path, pipe=pipe, bg=background)

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
