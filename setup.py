from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import tomli

def get_version():
    with open("pyproject.toml", "rb") as f:
        toml_data = tomli.load(f)
    
    return toml_data["project"]["version"]

ext_modules = [
    Pybind11Extension(
        "mcts_tsp._mcts_cpp",
        ["src/code/mcts.cpp"],
        define_macros=[("VERSION_INFO", get_version())],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
