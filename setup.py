from os import path
import glob
from setuptools import setup,Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CppExtension
import torch
import platform


def get_ext_cpp_source():
    this_dir = path.dirname(path.abspath(__file__))
    cpp_path = path.join('src','speedup','csrc')
    source = glob.glob(path.join(cpp_path, '**','*.cpp'), recursive=True)
    return source

setup(
    name='speedup',
    version='0.0.1',
    ext_modules=[
        CppExtension(
            name='speedup',
            sources=get_ext_cpp_source(),
            include_dirs=cpp_extension.include_paths(),
            language='c++'
        )
    ],
    packages=['src.speedup','src.speedup.compiler'],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False) if platform.system().lower() == 'windows' else torch.utils.cpp_extension.BuildExtension}
)
