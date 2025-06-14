import torch

from src.speedup.compiler.compile_config import CompileConfig
from src.speedup.compiler.speedup_backend import SpeedupBackend


def compile_model(model, config:CompileConfig):
    backend = SpeedupBackend(config)
    compiled_model = torch.compile(model, backend=backend)
    return compiled_model
