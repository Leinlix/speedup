import torch

from .compile_config import CompileConfig
from .speedup_backend import SpeedupBackend


def compile_model(model, config:CompileConfig):
    backend = SpeedupBackend(config)
    compiled_model = torch.compile(model, backend=backend)
    return compiled_model
