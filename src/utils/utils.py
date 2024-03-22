import math

import numpy as np
import torch
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset


def calc_row_idx(k, n):
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n, progress_bar=None):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    if progress_bar is not None:
        progress_bar.update(1)
    return i, j


def actual_indices(idx, n):
    n_row_elems = np.cumsum(np.arange(1, n)[::-1])
    ii = (n_row_elems[:, None] - 1 < idx[None, :]).sum(axis=0)
    shifts = np.concatenate([[0], n_row_elems])
    jj = np.arange(1, n)[ii] + idx - shifts[ii]
    if np.sum(ii < 0) > 0 or np.sum(jj < 0) > 0:
        logger.error("Negative indices")
    return ii, jj


def has_same_label(arr) -> np.ndarray:
    arr = np.array(arr)
    result = arr[:, None] == arr
    return result


def set_dataset_transformation(dataset: Dataset, transform: torch.nn.Module):
    def _set_transform(d: Dataset, transform: torch.nn.Module):
        if hasattr(d, "transforms"):

            def _transforms_wrapper(image, label):
                return transform(image), label

            d.transforms = _transforms_wrapper
        if hasattr(d, "transform"):
            d.transform = transform

    if type(dataset) is ConcatDataset:
        for d in dataset.datasets:
            _set_transform(d=d, transform=transform)
    else:
        _set_transform(d=dataset, transform=transform)
