import pathlib
from setuptools import setup
from setuptools import find_packages

wd = pathlib.Path(__file__).parent.resolve()


def get_requirements(path: str = ".") -> list:
    with open(f"{path}/requirements.txt") as f:
        requirements = f.read().splitlines()
        requirements = list(filter(lambda s: "=" in s, requirements))
        return requirements


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="finetuna",
    version="0.0.1",
    description="Fine tune LLMs.",
    long_description=long_description,
    url="https://github.com/MaximeRobeyns/llm_finetuner",
    license="GPLv3",
    install_requires=get_requirements(),
    packages=find_packages(exclude=[]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://maximerobeyns.github.io/llm_finetuner/",
        "Bug Reports": "https://github.com/MaximeRobeyns/llm_finetuner/issues",
        "Source": "https://github.com/MaximeRobeyns/llm_finetuner",
    },
)
