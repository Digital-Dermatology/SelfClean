import math

import numpy as np


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
        print("Negative indices")
    return ii, jj


def has_same_label(arr):
    arr = np.array(arr)
    result = arr[:, None] == arr
    return result
