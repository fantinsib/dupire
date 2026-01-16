r"""
 _ __  _   _ _ __ ___   ___ _ __(_) ___ __ _| |
| '_ \| | | | '_ ` _ \ / _ \ '__| |/ __/ _` | |
| | | | |_| | | | | | |  __/ |  | | (_| (_| | |
|_| |_|\__,_|_| |_| |_|\___|_|  |_|\___\__,_|_|                                    
"""
import math
import numpy as np
from typing import Optional

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def finite_difference(f_x: np.ndarray, x: np.ndarray, degree: int = 1, axis:int = 0) -> np.ndarray:
    """
    Derivative of f_x with respect to x

    Parameters
    ----------
    f_x : np.ndarray
        Array of shape (len(x), n) where the first axis matches x.
    x : np.ndarray
        1D grid (monotone increasing) of length f_x.shape[0].
    degree : int
        1 for first derivative, 2 for second derivative.
    axis: int
        The axis along which to compute the derivative.

    Returns
    -------
    np.ndarray
        Derivative array with the same shape as f_x.
    """

    if f_x.shape[axis] != len(x):
        raise ValueError("Array dimension does not match x along the chosen axis.")    
    f = np.moveaxis(f_x, axis, 0)
    d_f = np.empty_like(f)

    if degree == 1:
        d_f[1:-1, ...] = (f[2:, ...] - f[:-2, ...]) / (x[2:, None] - x[:-2, None])
        d_f[0, ...] = (f[1, ...] - f[0, ...]) / (x[1] - x[0])
        d_f[-1, ...] = (f[-1, ...] - f[-2, ...]) / (x[-1] - x[-2])

    elif degree == 2:
        dx_fwd = x[2:] - x[1:-1]
        dx_bwd = x[1:-1] - x[:-2]

        d_f[1:-1, ...] = 2.0 * (
            (f[2:, ...] - f[1:-1, ...]) / dx_fwd[:, None]
            - (f[1:-1, ...] - f[:-2, ...]) / dx_bwd[:, None]
        ) / (dx_fwd + dx_bwd)[:, None]

        d_f[0, ...] = d_f[1, ...]
        d_f[-1, ...] = d_f[-2, ...]

    else:
        raise ValueError("degree must be 1 or 2")

    return np.moveaxis(d_f, 0, axis)


