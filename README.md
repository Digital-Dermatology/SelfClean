# 🧼🔎 SelfClean

![SelfClean Teaser](https://github.com/Digital-Dermatology/SelfClean/raw/main/assets/SelfClean_Teaser.png)

A holistic self-supervised data cleaning strategy to detect irrelevant samples, near duplicates, and label errors.

**NOTE:** Make sure to have `git-lfs` installed before pulling the repository to ensure the pre-trained models are pulled correctly ([git-lfs install instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)).

This project is licensed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/).

<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="cc" width="20"/> <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="by" width="20"/> <img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="nc" width="20"/>

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

- <a href="https://github.com/fastai/imagenette">Imagenette</a> 🖼️
- <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a> 🐶
