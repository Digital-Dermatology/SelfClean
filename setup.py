import codecs
import os.path
import re
from os.path import abspath, dirname, join

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def parse_requirements(filename):
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()
PACKAGE_NAME = "selfclean"
SOURCE_DIRECTORY = "src"
SOURCE_PACKAGE_REGEX = re.compile(rf"^{SOURCE_DIRECTORY}")

source_packages = find_packages(include=[SOURCE_DIRECTORY, f"{SOURCE_DIRECTORY}.*"])
proj_packages = [
    SOURCE_PACKAGE_REGEX.sub(PACKAGE_NAME, name) for name in source_packages
]

setup(
    name=PACKAGE_NAME,
    packages=proj_packages,
    package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
    version="0.0.14",
    author="Fabian Groeger",
    author_email="fabian.groeger@unibas.ch",
    description="A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates and label errors.",
    long_description=README_MD,
    long_description_content_type="text/markdown",
    url="https://github.com/Digital-Dermatology/SelfClean",
    python_requires=">=3.6",
    install_requires=parse_requirements("requirements.txt"),
    setup_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
