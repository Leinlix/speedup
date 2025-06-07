import torch._inductor.pattern_matcher
from torch._inductor.pattern_matcher import PatternMatcherPass


def get_input():
    pass


def pattern():
    pass


def replacement():
    pass


class AddMultPass:
    def __init__(self):
        self.pattern : PatternMatcherPass = PatternMatcherPass(pass_name='AddMulPass')
        torch._inductor.pattern_matcher.register_replacement(pattern, replacement, get_input(), torch._inductor.pattern_matcher.fwd_only, self.pattern)


    def __call__(self, graph: torch.fx.GraphModule):
        self.pattern.apply(graph)