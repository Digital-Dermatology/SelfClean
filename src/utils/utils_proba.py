from enum import Enum

import numpy as np
import scipy


class DistPrior(Enum):
    FLAT = "Flat distribution"
    RANDOM = "Theoretical random distribution"


class Truncation(Enum):
    LOW = "Low truncation"
    DOUBLE = "Double truncation"


class FittingMode(Enum):
    DIRECT = "directly all parameters"
    AB_REST = "a, b then loc scale"
    AB_SCALE_LOC = "a, b then scale, then loc"


def log_truncated_beta_2par(ab, xq, x):
    a, b = ab
    # The normalization factor for a truncated beta distribution can be obtained
    # through a regularized incomplete beta function
    # Note that betainc(b, a, 1-x) = 1 - betainc(a, b, x)
    log_norm = np.log(scipy.special.betainc(b, a, 1 - xq)) + scipy.special.betaln(a, b)
    log_numerator = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
    return log_numerator - log_norm


def truncated_beta_2par(ab, xq, x):
    return np.exp(log_truncated_beta_2par(ab, xq, x))


def neg_log_truncated_beta_mean_2par(ab, xq, x):
    return -np.mean(log_truncated_beta_2par(ab, xq, x))


def log_truncated_beta_3par(pars, xq, x):
    a, b, scale = pars
    t = x / scale
    tq = xq / scale
    # The normalization factor for a truncated beta distribution can be obtained
    # through a regularized incomplete beta function
    # Note that betainc(b, a, 1-t) = 1 - betainc(a, b, t)
    log_norm = np.log(scipy.special.betainc(b, a, 1 - tq)) + scipy.special.betaln(a, b)
    log_numerator = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    log_norm += np.log(scale)
    return log_numerator - log_norm


def truncated_beta_3par(pars, xq, x):
    return np.exp(log_truncated_beta_3par(pars, xq, x))


def neg_log_truncated_beta_mean_3par(pars, xq, x):
    return -np.mean(log_truncated_beta_3par(pars, xq, x))


def log_truncated_beta_4par(pars, xq, x):
    a, b, loc, scale = pars
    t = (x - loc) / scale
    tq = (xq - loc) / scale
    # The normalization factor for a truncated beta distribution can be obtained
    # through a regularized incomplete beta function
    # Note that betainc(b, a, 1-t) = 1 - betainc(a, b, t)
    log_norm = np.log(scipy.special.betainc(b, a, 1 - tq)) + scipy.special.betaln(a, b)
    log_numerator = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    log_norm += np.log(scale)
    return log_numerator - log_norm


def truncated_beta_4par(pars, xq, x):
    return np.exp(log_truncated_beta_4par(pars, xq, x))


def neg_log_truncated_beta_mean_4par(pars, xq, x):
    return -np.mean(log_truncated_beta_4par(pars, xq, x))


def log_double_truncated_beta_2par(ab, xqlow, xqhigh, x):
    a, b = ab
    log_norm = np.log(
        scipy.special.betainc(a, b, xqhigh) - scipy.special.betainc(a, b, xqlow)
    )
    log_norm += scipy.special.betaln(a, b)
    log_numerator = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
    return log_numerator - log_norm


def double_truncated_beta_2par(ab, xqlow, xqhigh, x):
    return np.exp(log_double_truncated_beta_2par(ab, xqlow, xqhigh, x))


def neg_log_double_truncated_beta_mean_2par(ab, xqlow, xqhigh, x):
    return -np.mean(log_double_truncated_beta_2par(ab, xqlow, xqhigh, x))


def log_double_truncated_beta_3par(pars, xqlow, xqhigh, x):
    a, b, scale = pars
    t = x / scale
    tqlow = xqlow / scale
    tqhigh = xqhigh / scale
    log_norm = (
        np.log(scipy.special.betainc(a, b, tqhigh) - scipy.special.betainc(a, b, tqlow))
        + scipy.special.betaln(a, b)
        + np.log(scale)
    )
    log_numerator = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    return log_numerator - log_norm


def double_truncated_beta_3par(pars, xqlow, xqhigh, x):
    return np.exp(log_double_truncated_beta_3par(pars, xqlow, xqhigh, x))


def neg_log_double_truncated_beta_mean_3par(pars, xqlow, xqhigh, x):
    return -np.mean(log_double_truncated_beta_3par(pars, xqlow, xqhigh, x))


def log_double_truncated_beta_4par(pars, xqlow, xqhigh, x):
    a, b, loc, scale = pars
    t = (x - loc) / scale
    tqlow = (xqlow - loc) / scale
    tqhigh = (xqhigh - loc) / scale
    log_norm = (
        np.log(scipy.special.betainc(a, b, tqhigh) - scipy.special.betainc(a, b, tqlow))
        + scipy.special.betaln(a, b)
        + np.log(scale)
    )
    log_numerator = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    return log_numerator - log_norm


def double_truncated_beta_4par(pars, xqlow, xqhigh, x):
    return np.exp(log_double_truncated_beta_4par(pars, xqlow, xqhigh, x))


def neg_log_double_truncated_beta_mean_4par(pars, xqlow, xqhigh, x):
    return -np.mean(log_double_truncated_beta_4par(pars, xqlow, xqhigh, x))


def get_scale_loc(dist, x, q1, q2):
    x1 = np.quantile(x, q1)
    x2 = np.quantile(x, q2)
    y1 = dist.ppf(q1)
    y2 = dist.ppf(q2)
    scale = (x1 - x2) / (y1 - y2)
    loc = (y1 * x2 - y2 * x1) / (y1 - y2)
    return scale, loc
