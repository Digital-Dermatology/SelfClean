import codecs
import os.path
from os.path import abspath, dirname, join

from setuptools import find_namespace_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def parse_requirements(filename):
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

setup(
    name="selfclean",
    version="0.0.1",
    author="Fabian Groeger",
    author_email="fabian.groeger@bluewin.ch",
    description="A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates and label errors.",
    long_description=README_MD,
    long_description_content_type="text/markdown",
    url="https://github.com/Digital-Dermatology/SelfClean",
    packages=find_namespace_packages(),
    keywords="data-cleaning, self-supervised-learning, data-centric-ai",
    install_reqs=parse_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
)
