# 🧼🔎 SelfClean

[![Test and Coverage](https://github.com/Digital-Dermatology/SelfClean/actions/workflows/pytest-coverage.yml/badge.svg)](https://github.com/Digital-Dermatology/SelfClean/actions/workflows/pytest-coverage.yml)

![SelfClean Teaser](https://github.com/Digital-Dermatology/SelfClean/raw/main/assets/SelfClean_Teaser.png)

A holistic self-supervised data cleaning strategy to detect off-topic samples, near duplicates, and label errors.

**Publications:** [SelfClean Paper (NeurIPS24)](https://arxiv.org/abs/2305.17048) | [Data Cleaning Protocol Paper (ML4H23@NeurIPS)](https://arxiv.org/abs/2309.06961)

**NOTE:** Make sure to have `git-lfs` installed before pulling the repository to ensure the pre-trained models are pulled correctly ([git-lfs install instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)).

This project is licensed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/).

<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="cc" width="20"/> <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="by" width="20"/> <img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="nc" width="20"/>

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

selfclean = SelfClean(
    # displays the top-7 images from each error type
    # per default this option is disabled
    plot_top_N=7,
)

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
df_off_topic_samples = issues.get_issues("off_topic_samples", return_as_df=True)
df_label_errors = issues.get_issues("label_errors", return_as_df=True)
```

**Examples:**
In `examples/`, we've provided some example notebooks in which you will learn how to analyze and clean datasets using SelfClean.
These examples analyze different benchmark datasets such as:

- <a href="https://github.com/fastai/imagenette">Imagenette</a> 🖼️ (Open in <a href="https://nbviewer.org/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">NBViewer</a> | <a href="https://github.com/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">GitHub</a> | <a href="https://colab.research.google.com/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_Imagenette.ipynb">Colab</a>)
- <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a> 🐶 (Open in <a href="https://nbviewer.org/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">NBViewer</a> | <a href="https://github.com/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">GitHub</a> | <a href="https://colab.research.google.com/github/Digital-Dermatology/SelfClean/blob/main/examples/Investigate_OxfordIIITPet.ipynb">Colab</a>)

Also, check out our <a href="https://www.kaggle.com/code/fabiangrger/removing-the-psychic-from-the-dataset">Kaggle notebook</a> to see an illustration of how to get a gold medal for cleaning a competition dataset.

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
