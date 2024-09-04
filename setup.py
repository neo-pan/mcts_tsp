from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "mcts_tsp._mcts_cpp",
        ["src/code/mcts.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="mcts_tsp",
    version=__version__,
    author="Xuanhao Pan",
    author_email="xuanhaopan@link.cuhk.edu.cn",
    description="A python wrapper for MCTS TSP solver",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
)