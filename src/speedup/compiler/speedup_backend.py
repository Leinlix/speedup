import torch
from typing import List, Callable


class SpeedupBackend:
    def __init__(self, passes: List[Callable[torch.fx.graph], None]):
        self.compile_passes = passes
        self.induct_compile_config = dict()
        self.induct_compile_config['post_grad_custom_post_pass'] = self.post_pass_run


    def __call__(self, graph : torch.fx.GraphModule, example_input : List[torch.Tensor]):
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph, example_input, config_patches=self.induct_compile_config)


    def post_pass_run(self, graph: torch.fx.graph):
        for p in self.compile_passes:
            p(graph)