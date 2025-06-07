import torch

from src.speedup.compiler.compile_config import CompileConfig
from src.speedup.compiler.speedup_backend import SpeedupBackend


def compile_model(model, config:CompileConfig):
    passes = config.pass_list
    backend = SpeedupBackend
    torch.compile(model, backend=backend)