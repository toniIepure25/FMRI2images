"""Fallback setup script for fmri2img (use pyproject.toml for primary installs)."""
from setuptools import setup, find_packages

setup(
    name="fmri2img",
    version="2.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
