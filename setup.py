from setuptools import setup, find_packages

setup(
    name="RCP_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spikeinterface",
        "numpy",
        "scipy",
    ],
)