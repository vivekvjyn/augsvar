import numpy as np
from tsaug import TimeWarp, Drift, Resize

def bounds(ts):
    """Get the bounds of the non-NaN values in the time series.

    Args:
        ts (list or numpy.ndarray): The time series.

    Returns:
        tuple: The start and end indices of the non-NaN values.
    """
    idx = np.where(~np.isnan(np.array(ts)))[0]
    a = min(idx) if len(idx) else 0
    b = max(idx) if len(idx) else 0
    return a, b

def perturb(ts, n_speed_change, max_drift, proportion):
    """Perturb time series.

    Args:
        ts (list or numpy.ndarray): The time series.
        n_speed_change (int): The number of speed changes.
        max_drift (float): The maximum drift.
        proportion (float): The proportion of the time series to resize.

    Returns:
        numpy.ndarray: The perturbed time series.
    """
    ab = bounds(ts)
    x = np.array(ts[:ab[0]])
    y = np.array(ts[ab[0]: ab[1]])
    z = np.array(ts[ab[1]:])
    if len(y) > 4:
        y = TimeWarp(n_speed_change=n_speed_change).augment(y)
        y = Drift(max_drift=max_drift).augment(y)
    y = Resize(max(1, int(round(len(ts) * np.random.uniform(1 - proportion, 1 + proportion))))).augment(y)
    return np.concatenate([x, y, z])

def augment(x, y, z, n_speed_change, max_drift, proportion):
    """Apply augmentation to time series.

    Args:
        x (list or numpy.ndarray): The previous time series.
        y (list or numpy.ndarray): The current time series.
        z (list or numpy.ndarray): The successor time series.

    Returns:
        tuple: The augmented time series.
    """
    x_new = perturb(x, n_speed_change=n_speed_change, max_drift=max_drift, proportion=proportion)
    y_new = perturb(y, n_speed_change=n_speed_change, max_drift=max_drift, proportion=proportion)
    z_new = perturb(z, n_speed_change=n_speed_change, max_drift=max_drift, proportion=proportion)
    return x_new, y_new, z_new
