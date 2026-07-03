from setuptools import find_packages, setup

setup(
    name="ree",
    version="0.1.0",
    description="Estimating rare events in language models with proper evaluation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.1",
        "numpy",
        "tqdm",
        "matplotlib",
        "datasets>=2.17.0",
        "transformers",
        "huggingface-hub<=0.22.2",
        "jaxtyping",
        "fancy-einsum",
        "blobfile",
    ],
)
