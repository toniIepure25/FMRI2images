"""Setup script for fmri2img package (fallback if pyproject.toml fails)"""
from setuptools import setup, find_packages

setup(
    name="fmri2img",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
