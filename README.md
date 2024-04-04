# 🧼🔎 SelfClean 

[![Test and Coverage](https://github.com/Digital-Dermatology/SelfClean/actions/workflows/pytest-coverage.yml/badge.svg)](https://github.com/Digital-Dermatology/SelfClean/actions/workflows/pytest-coverage.yml)

![SelfClean Teaser](https://github.com/Digital-Dermatology/SelfClean/raw/main/assets/SelfClean_Teaser.png)

A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates, and label errors.

**Publications:** [SelfClean Paper](https://arxiv.org/abs/2305.17048) | [Data Cleaning Protocol Paper (ML4H24@NeurIPS)](https://arxiv.org/abs/2309.06961)

**NOTE:** Make sure to have `git-lfs` installed before pulling the repository to ensure the pre-trained models are pulled correctly ([git-lfs install instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)).

## Installation

> Install SelfClean via [PyPI](https://pypi.org/project/selfclean/):

```python
# upgrade pip to its latest version
pip install -U pip

# install selfclean
pip install selfclean

# Alternatively, use explicit python version (XX)
python3.XX -m pip install selfclean
```

## Getting Started

You can run SelfClean in a few lines of code:

```python
from selfclean import SelfClean

selfclean = SelfClean()

# run on pytorch dataset
issues = selfclean.run_on_dataset(
    dataset=copy.copy(dataset),
)
# run on image folder
issues = selfclean.run_on_image_folder(
    input_path="path/to/images",
)

# get the data quality issue rankings
df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
df_irrelevants = issues.get_issues("irrelevants", return_as_df=True)
df_label_errors = issues.get_issues("label_errors", return_as_df=True)
```

**Examples:**
In `examples/`, we've provided some example notebooks in which you will learn how to analyze and clean datasets using SelfClean.
These examples analyze different benchmark datasets such as:

- <a href="https://github.com/fastai/imagenette">Imagenette</a> 🖼️ (Open in <a href="https://nbviewer.org/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">NBViewer</a> | <a href="https://github.com/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">GitHub</a> | <a href="https://colab.research.google.com/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">Colab</a>)
- <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a> 🐶 (Open in <a href="https://nbviewer.org/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">NBViewer</a> | <a href="https://github.com/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">GitHub</a> | <a href="https://colab.research.google.com/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">Colab</a>)

## Development Environment
Run `make` for a list of possible targets.

Run these commands to install the requirements for the development environment:
```bash
make init
make install
```

To run linters on all files:
```bash
pre-commit run --all-files
```

We use the following packages for code and test conventions:
- `black` for code style
- `isort` for import sorting
- `pytest` for running tests
