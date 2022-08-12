import re
import pathlib
from setuptools import setup
from setuptools import find_packages

wd = pathlib.Path(__file__).parent.resolve()


def get_requirements(path: str = ".") -> list:
    with open(f"{path}/requirements.txt") as f:
        requirements = f.read().splitlines()
        requirements = list(filter(lambda s: "=" in s, requirements))
        return requirements


def get_deps(path: str = ".") -> list:
    deps = []
    with open(f"{path}/requirements.txt") as f:
        for line in f:
            s = re.search(r"^--find-links (?P<link>.*)$", line)
            if s is not None:
                deps.append(s.group("link"))
    return deps


install_requires, docs_requires, tests_requires = map(
    get_requirements, (".", "docs", "tests")
)

deps = [d for dd in [get_deps(p) for p in (".", "docs")] for d in dd]


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="finetuna",
    version="0.0.1",
    description="Efficiently fine-tune large pretrained models.",
    long_description=long_description,
    url="https://github.com/MaximeRobeyns/llm_finetuner",
    license="GPLv3",
    install_requires=install_requires,
    extras_require={
        "tests": tests_requires,
        "docs": docs_requires,
        "all": docs_requires,
    },
    dependency_links=deps,
    packages=find_packages(exclude=["docs", "tests"]),
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
