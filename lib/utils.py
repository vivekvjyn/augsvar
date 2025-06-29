import numpy as np
from tsaug import TimeWarp, Drift, Resize

def get_bounds(ts):
    idx = np.where(~np.isnan(np.array(ts)))[0]

    a = min(idx) if len(idx) else 0
    b = max(idx) if len(idx) else 0

    return a, b

def timewarp(ts):
    timewarped = TimeWarp(n_speed_change=5).augment(ts)
    return timewarped

def drift(ts):
    drifted = Drift(max_drift=0.01).augment(ts)

    return drifted

def resize(ts):
    proportion = np.random.uniform(0.95, 1.05)
    new_length = max(1, int(round(len(ts) * proportion)))
    resized = Resize(new_length).augment(ts)

    return resized

def preturb(ts):
    bounds = get_bounds(ts)

    a = np.array(ts[:bounds[0]])
    b = np.array(ts[bounds[0]: bounds[1]])
    c = np.array(ts[bounds[1]:])

    if len(b) > 4:
        b = timewarp(b)
        b = drift(b)

    b = resize(b)

    return np.concatenate([a, b, c])
