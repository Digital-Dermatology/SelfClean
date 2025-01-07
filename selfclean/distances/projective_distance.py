from typing import Optional

import numpy as np


def pairwise_projective_distance(X: np.ndarray, Y: Optional[np.ndarray] = None):
    r"""
    Pairwise projective distance.

    $$
    d(\vec{a},\vec{b}) = \frac{1}{\sqrt{2}}
    \min\left[
    \left|\frac{\vec{a}}{|\vec{a}|}-\frac{\vec{b}}{|\vec{b}|}\right|,
    \left|\frac{\vec{a}}{|\vec{a}|}+\frac{\vec{b}}{|\vec{b}|}\right|
    \right]
    $$
    """
    X = np.array(X)
    if Y is not None:
        Y = np.array(Y)
    else:
        X = Y

    mag_X = np.linalg.norm(X, axis=1)
    mag_Y = np.linalg.norm(Y, axis=1)

    norm_X = X / mag_X[:, np.newaxis]
    norm_Y = Y / mag_Y[:, np.newaxis]

    term1 = np.linalg.norm(norm_X[:, np.newaxis] - norm_Y, axis=-1)
    term2 = np.linalg.norm(norm_X[:, np.newaxis] + norm_Y, axis=-1)

    distance = np.minimum(term1, term2) / np.sqrt(2)

    return distance
