from setuptools import setup, find_packages

setup(
    name="gfn-od",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "hydra-core",
        "wandb",
        "tqdm",
    ],
    python_requires=">=3.8",
)