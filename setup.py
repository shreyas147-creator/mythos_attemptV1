"""Setup script for Mythos Transformer."""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mythos-transformer",
    version="1.0.0",
    author="Your Organization",
    author_email="team@your-org.com",
    description="Production implementation of Mythos-level transformer architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mythos-transformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "pytest>=8.0.0",
            "mypy>=1.9.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mythos-train=src.training.trainer:main",
            "mythos-eval=src.evaluation.evaluator:main",
        ],
    },
)
