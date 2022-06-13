from setuptools import find_packages, setup

setup(
    name="go_explore",
    packages=find_packages(),
    url="https://github.com/qgallouedec/go-explore",
    description="Implementation of Go-Explore based on stable-baselines3",
    long_description=open("README.md").read(),
    install_requires=[
        "stable_baselines3 @ git+https://git@github.com/qgallouedec/stable-baselines3@lge",
        "torchvision",
        "gym[atari, accept-rom-license]",
        "ale-py==0.7.4",
        "optuna",
        "opencv-python",
    ],
    extras_require={
        "tests": ["pytest", "black", "isort"],
    },
)
