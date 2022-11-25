from setuptools import find_packages, setup

setup(
    name="lge",
    packages=find_packages(),
    url="https://github.com/qgallouedec/go-explore",
    description="Implementation of Go-Explore based on stable-baselines3",
    long_description=open("README.md").read(),
    install_requires=[
        "stable_baselines3 @ git+https://git@github.com/qgallouedec/stable-baselines3@IM_and_Vec_HER",
        "importlib-metadata==4.13.0",
    ],
    extras_require={
        "tests": ["pytest", "black", "isort"],
        "experiments": [
            "optuna",
            "panda_gym @ git+https://github.com/qgallouedec/panda-gym.git@no-task",
            "gym_continuous_maze @ git+https://github.com/qgallouedec/gym-continuous-maze.git",
        ],
    },
)
