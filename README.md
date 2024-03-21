# SelfClean

[**SelfClean Paper**](https://arxiv.org/abs/2305.17048) | [**Data Cleaning Protocol Paper**](https://arxiv.org/abs/2309.06961)

<p align="center">
  <img src="https://raw.githubusercontent.com/Digital-Dermatology/SelfClean/main/assets/SelfClean_Teaser.svg?token=GHSAT0AAAAAACMMIN54Q4RK6OKJQ4AHE5XQZP4B5VQ">
</p>

<h2 align="center">

[![PyPI version](https://badge.fury.io/py/selfclean.svg)](https://badge.fury.io/py/selfclean)
![Contribotion](https://img.shields.io/badge/Contribution-Welcome-brightgreen)

</h2>

A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates, and label errors.

## Development Environment
Run `make` for a list of possible targets.

### Installation
Run these commands to install the project:
```bash
make init
make install
```

To run linters on all files:
```bash
pre-commit run --all-files
```

### Code and test conventions
- `black` for code style
- `isort` for import sorting
- `pytest` for running tests
