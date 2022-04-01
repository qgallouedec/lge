from setuptools import setup, find_packages

setup(
    name="go_explore",
    packages=find_packages(),
    url="https://github.com/qgallouedec/go-explore",
    description="Implementation of Go-Explore based on stable-baselines3",
    long_description=open("README.md").read(),
    install_requires=[
        "stable_baselines3 @ git+https://git@github.com/qgallouedec/stable-baselines3@action-repeat",
        "torchvision",
        "gym[atari, accept-rom-license]",
        "optuna",
    ],
    extras_require={
        "tests": ["pytest", "black", "isort"],
    },
)
