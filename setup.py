from setuptools import setup, Extension
import pybind11

# Lista de todos tus archivos .cpp fuente
cpp_files = [
    "src/bindings.cpp",
    "src/tensor.cpp",
    "src/utils.cpp", 
    "src/core.cpp",
    "src/loss.cpp",
    "src/ops.cpp",
    "src/optimizers.cpp",
    "src/neuron.cpp",
    "src/trainer.cpp",
    "src/CPUBackend.cpp"
    # EXCEPTO main.cpp (no queremos un ejecutable, sino una librería)
]

ext_modules = [
    Extension(
        "learntorch", # Nombre del paquete en Python
        sorted(cpp_files),
        include_dirs=[
            pybind11.get_include(),
            "include"
        ],
        language="c++",
        extra_compile_args=["-std=c++17"], # Asegura estándar moderno
    ),
]

setup(
    name="learntorch",
    version="0.1",
    ext_modules=ext_modules,
)