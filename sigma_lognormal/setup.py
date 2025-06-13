from setuptools import setup, find_packages

setup(
    name="sigma_lognormal",
    version="0.1.0",
    description="Handstroke‐extraction library for signatures",
    author="Ritvik Shrivastava",
    packages=find_packages(),  # this will include all submodules
    install_requires=[
        "numpy",
        "scipy",
        # runtime dependencies (e.g. “torch”, “matplotlib”) if not installed elsewhere
    ],
)
