# SelfClean
A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates and label errors.

<p align="center">
  <img src="assets/SelfClean_Teaser.svg">
</p>

[**SelfClean Paper**](https://arxiv.org/abs/2305.17048)


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
