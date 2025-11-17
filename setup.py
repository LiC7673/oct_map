from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
setup(
    name='oct_map',  # 你在 Python 中 import 时用的名字
    ext_modules=[
        CUDAExtension(
            # 模块的C扩展名，必须和 setup.py 里的 'name' 匹配
            name="oct_map",

            # 1. 你的所有 *源文件* (.cpp, .cu)
            # .h 头文件不需要列在这里，编译器会自动找到它们
            sources=['tree.cu',"ext.cpp","utils.cu"],

            # 2. (可选) 额外的编译参数
            # extra_compile_args={'cxx': ['-O3'],
            #                     'nvcc': ['-O3', '--use_fast_math']})
        extra_compile_args = {"nvcc": [], "cxx": cxx_compiler_flags})

    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)