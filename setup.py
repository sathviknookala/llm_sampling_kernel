import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = sorted(glob.glob("csrc/*.cpp") + glob.glob("csrc/*.cu"))
headers = sorted(glob.glob("csrc/*.cuh"))

setup(
    name="fused_sampling",
    ext_modules=[
        CUDAExtension(
            name="fused_sampling",
            sources=sources,
            depends=headers,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # no --use_fast_math: Gate A compares renormalized probabilities against an
                # fp32 reference, and __expf would put the kernel out of reach of that
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
