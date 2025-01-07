import math
from functools import partial
from pathlib import Path

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


def triu_indices_memmap(filename: str, N: int, k: int = 0):
    """
    Generate the indices for the upper-triangular part of a matrix using memmap.

    Parameters:
    filename (str): The name of the file to use for memmap.
    N (int): The size of the square matrix.
    k (int): Diagonal offset. k=0 is the main diagonal, k>0 is above, and k<0 is below.

    Returns:
    tuple of ndarray: Indices for the upper-triangular part of the matrix.
    """
    # Calculate the number of elements in the upper triangular part
    num_elements = sum(max(0, N - k - i) for i in range(N))

    # Create memmap arrays for row and column indices
    rows_filename = Path(filename + "_rows.dat")
    cols_filename = Path(filename + "_cols.dat")
    if rows_filename.exists():
        rows_filename.unlink()
    if cols_filename.exists():
        cols_filename.unlink()
    rows_memmap = np.memmap(
        str(rows_filename),
        dtype="int64",
        mode="w+",
        shape=(num_elements,),
    )
    cols_memmap = np.memmap(
        str(cols_filename),
        dtype="int64",
        mode="w+",
        shape=(num_elements,),
    )

    idx = 0
    for i in range(N):
        for j in range(i + k, N):
            rows_memmap[idx] = i
            cols_memmap[idx] = j
            idx += 1

    return rows_memmap, cols_memmap


def has_same_label(arr) -> np.ndarray:
    arr = np.array(arr)
    result = arr[:, None] == arr
    return result


def _transforms_wrapper(transform, image, label):
    return transform(image), label


def _set_transform(d: Dataset, transform: torch.nn.Module):
    if hasattr(d, "transforms"):
        _transform = partial(_transforms_wrapper, transform=transform)
        d.transforms = _transform
    if hasattr(d, "transform"):
        d.transform = transform


def set_dataset_transformation(dataset: Dataset, transform: torch.nn.Module):
    if type(dataset) is ConcatDataset:
        for d in dataset.datasets:
            _set_transform(d=d, transform=transform)
    else:
        _set_transform(d=dataset, transform=transform)
