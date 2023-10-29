import pathlib
from setuptools import setup, find_packages


# Get the long description from the README file
readme_path = pathlib.Path(__file__).parent.resolve() / "README.md"
long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="torch-imle",
    version="1.0.0",
    description="Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uclnlp/torch-imle",
    author="UCL Natural Language Processing",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="torch, deep learning, machine learning, gradient estimation",
    packages=find_packages(),
    python_requires=">=3.9, <4",
    install_requires=["torch"],
    project_urls={
        "Bug Reports": "https://github.com/uclnlp/torch-imle/issues",
        "Source": "https://github.com/uclnlp/torch-imle",
        "Paper": "https://arxiv.org/abs/2106.01798",
    },
)
