"""Setup script for Deep-PrAE package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepprae",
    version="1.0.0",
    author="Deep-PrAE Authors",
    author_email="",
    description="Deep Probabilistic Rare Event Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Deep-PRAE",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "torch>=1.7.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "pyomo>=6.0.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=20.8b1",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "solvers": [
            "gurobipy>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepprae-run=run_all_examples:main",
        ],
    },
)
