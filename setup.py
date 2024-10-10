from setuptools import setup
import os
import os.path as osp
import warnings

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('epcq/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []
include_dirs = [osp.join(ROOT_DIR, "epcq", "csrc", "include")]

# From PyTorch3D
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
	prefix = os.environ.get("CONDA_PREFIX", None)
	if prefix is not None and os.path.isdir(prefix + "/include/cub"):
		cub_home = prefix + "/include"

if cub_home is None:
	warnings.warn(
		"The environment variable `CUB_HOME` was not found."
		"Installation will fail if your system CUDA toolkit version is less than 11."
		"NVIDIA CUB can be downloaded "
		"from `https://github.com/NVIDIA/cub/releases`. You can unpack "
		"it to a location of your choice and set the environment variable "
		"`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
	)
else:
	include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))


try:
    ext_modules = [
        CUDAExtension('epcq.csrc', [
            'epcq/csrc/epcq.cpp',
            'epcq/csrc/rasterization.cu',
            'epcq/csrc/optimize.cu',
            'epcq/csrc/epcq_render_topk.cu',
            'epcq/csrc/epcq_render_topk_bk.cu'

        ], include_dirs=include_dirs,
        optional=False),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='epcq',
    version=__version__,
    author='Jiahao Ma',
    author_email='jiahao.ma@anu.edu.au',
    description='Efficient Point Cloud Query with CUDA',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['epcq', 'epcq.csrc'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
