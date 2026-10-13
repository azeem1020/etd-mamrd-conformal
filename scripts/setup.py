from setuptools import find_packages, setup

setup(
    name="phased",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    python_requires=">=3.9",
)
