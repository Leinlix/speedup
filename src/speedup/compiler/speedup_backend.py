import torch
from typing import List, Callable
from .compile_config import CompileConfig

class SpeedupBackend:
    def __init__(self, config: CompileConfig):
        self.config = config
        self.induct_compile_config = dict()
        self.induct_compile_config['post_grad_custom_pre_pass'] = self.pre_pass_run
        self.induct_compile_config['post_grad_custom_post_pass'] = self.post_pass_run


    def __call__(self, graph : torch.fx.GraphModule, example_input : List[torch.Tensor]):
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph, example_input, config_patches=self.induct_compile_config)


    def pre_pass_run(self, graph: torch.fx.graph):
        if self.config.verbose:
            print('---------------- pre pass run -----------------')
            print(graph)
        for p in self.config.pre_pass_list:
            p(graph)


    def post_pass_run(self, graph: torch.fx.graph):
        if self.config.verbose:
            print('---------------- post pass run -----------------')
            print(graph)
        for p in self.config.post_pass_list:
            p(graph)