import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # n_output_dims: int = 3
        # height: int = 1024
        # width: int = 1024
        # color_activation: str = "sigmoid"
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, dirs: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        raise NotImplementedError
