import platform

"""
this class defines the compile option

use_triton : only support linux
use_xformer
"""
class CompileConfig:
    use_triton = True
    def __init__(self, use_default_pass=True, verbose=False):
        self.verbose = verbose
        self.pre_pass_list = []
        self.post_pass_list = []
        if use_default_pass:
            self.pre_pass_list  = []
            self.post_pass_list = []

