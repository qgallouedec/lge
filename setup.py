from setuptools import setup, find_packages

setup(
    name="go_explore",
    packages=find_packages(),
    url="https://github.com/qgallouedec/go-explore",
    description="Implementation of Go-Explore",
    long_description=open("README.md").read(),
    install_requires=[
        "stable_baselines3 @ git+https://git@github.com/qgallouedec/stable-baselines3@ICM-compat",
        "scipy",
        "panda-gym @ git+https://git@github.com/qgallouedec/panda-gym@no-task",
    ],
    extras_require={
        "tests": ["pytest", "black"],
        "extra": ["optuna"],
    },
)
