import os
import pickle5 as pickle
import json
import yaml
import csv
import logging
import math

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d, make_interp_spline
from scipy.signal import argrelextrema, savgol_filter, resample

def get_logger(name):
    logging.basicConfig(format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger(__name__)

def load_json(path):
    """
    Load json at <path> to dict

    :param path: path of json
    :type path: str

    :return: dict of json information
    :rtype: dict
    """
    # Opening JSON file
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(j, path):
    """
    Write json, <j>, to <path>

    :param j: json
    :type path: json
    :param path: path to write to,
        if the directory doesn't exist, one will be created
    :type path: str
    """
    create_if_not_exists(path)
    # Opening JSON file
    with open(path, 'w') as f:
        json.dump(j, f)


def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def load_annotations(path, delim='\t'):
    vals = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter=delim, quotechar='"')
        for l,_,s,e,d,a in rd:
            vals.append([str(l), s, e, d, str(a)])

    df = pd.DataFrame(vals, columns=['type', 'start', 'end', 'duration', 'label'])

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    df['start_sec'] = df['start'].dt.microsecond*0.000001 + df['start'].dt.second + df['start'].dt.minute*60
    df['end_sec'] = df['end'].dt.microsecond*0.000001 + df['end'].dt.second + df['end'].dt.minute*60

    df['label'] = df['label'].apply(lambda y: y.strip().lower())

    return df[['start', 'end', 'start_sec', 'end_sec', 'label']]

def load_yaml(path):
    """
    Load yaml at <path> to dictionary, d

    Returns
    =======
    Wrapper dictionary, D where
    D = {filename: d}
    """
    import zope.dottedname.resolve
    def constructor_dottedname(loader, node):
        value = loader.construct_scalar(node)
        return zope.dottedname.resolve.resolve(value)

    def constructor_paramlist(loader, node):
        value = loader.construct_sequence(node)
        return ParamList(value)

    yaml.add_constructor('!paramlist', constructor_paramlist)
    yaml.add_constructor('!dottedname', constructor_dottedname)

    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d


def cpath(*args):
    """
    Wrapper around os.path.join, create path concatenating args and
    if the containing directories do not exist, create them.
    """
    path = os.path.join(*args)
    create_if_not_exists(path)
    return path


def write_pitch_track(pitch_track, path, sep='\t'):
    """
    Write pitch contour to tsv at <path>
    """
    with open(path,'w') as file:
        for t, p in pitch_track:
            file.write(f"{t}{sep}{p}")
            file.write('\n')


def load_pitch_track(path, delim='\t'):
    """
    load pitch contour from tsv at <path>

    :param path: path to load pitch contour from
    :type path: str

    :return: Two numpy arrays of time and pitch values
    :rtype: tuple(numpy.array, numpy.array)
    """
    pitch_track = []
    with open(path) as fd:
        rd = csv.reader(fd, delimiter=delim, quotechar='"')
        for t,p in rd:
            pitch_track.append([float(t),float(p)])

    return np.array(pitch_track)




def pitch_to_cents(p, tonic):
    """
    Convert pitch value, <p> to cents above <tonic>.

    :param p: Pitch value in Hz
    :type p: float
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <p> in cents above <tonic>
    :rtype: float
    """
    return 1200*math.log(p/tonic, 2) if p else None


def cents_to_pitch(c, tonic):
    """
    Convert cents value, <c> to pitch in Hz

    :param c: Pitch value in cents above <tonic>
    :type c: float/int
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <c> in Hz
    :rtype: float
    """
    return (2**(c/1200))*tonic


def cents_seq_to_pitch(cseq, tonic):
    """
    Convert sequence of cents above <tonic> to sequence of
    pitch values in Hz

    :param cseq: Array of pitch values in cents above <tonic>
    :type cseq: np.array
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Sequence of pitch values in Hz
    :rtype: np.array
    """
    return np.vectorize(lambda y: cents_to_pitch(y, tonic))(cseq)

def pitch_seq_to_cents(pseq, tonic):
    """
    Convert sequence of pitch values to sequence of
    cents above <tonic> values

    :param pseq: Array of pitch values in Hz
    :type pseq: np.array
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Sequence of original pitch value in cents above <tonic>
    :rtype: np.array
    """
    return np.vectorize(lambda y: pitch_to_cents(y, tonic))(pseq)

def get_plot_kwargs(raga, tonic, cents=False, svara_cent_path = "conf/svara_cents.yaml", svara_freq_path = "conf/svara_lookup.yaml"):
    svara_cent = load_yaml(svara_cent_path)
    svara_freq = load_yaml(svara_freq_path)

    arohana = svara_freq[raga]['arohana']
    avorahana = svara_freq[raga]['avarohana']
    all_svaras = list(set(arohana+avorahana))

    if not cents:
        svara_cent = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}

    yticks_dict = {k:v for k,v in svara_cent.items() if any([x in k for x in all_svaras])}

    return {
        'yticks_dict':yticks_dict,
        'cents':cents,
        'tonic':tonic,
        'emphasize':['S', 'S ', 'S  ', ' S', '  S'],
        'figsize':(15,4)
    }


def subsample_series(time_series, pitch_series, proportion):
    """
    Subsample both time and pitch series by a proportion.

    Parameters:
    time_series (array-like): The time values corresponding to the pitch values.
    pitch_series (array-like): The pitch values to be subsampled.
    proportion (float): Proportion of data to retain. Must be between 0 and 1.

    Returns:
    subsampled_time (array-like): The subsampled time series.
    subsampled_pitch (array-like): The subsampled pitch series.
    """
    if not (0 < proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1 (exclusive of 0).")

    # Calculate the number of points to keep
    total_points = len(time_series)
    num_to_keep = int(np.floor(total_points * proportion))

    if num_to_keep == 0:
        raise ValueError("Proportion too small, results in zero points being kept.")

    # Generate indices that are evenly spaced based on the proportion
    indices = np.linspace(0, total_points - 1, num_to_keep, dtype=int)

    # Subsample both arrays using these indices
    subsampled_time = np.array(time_series)[indices]
    subsampled_pitch = np.array(pitch_series)[indices]

    return subsampled_time, subsampled_pitch



def smooth_pitch_curve(time_series, pitch_series, smoothing_factor=0.6, min_points=4):
    """
    Smooth a quantized pitch time series in contiguous chunks using cubic splines,
    while handling None/NaN values and maintaining critical features. The data is
    normalized to the range 0-1 before smoothing, and rescaled back to its original
    range afterward.

    Parameters:
    time_series (array-like): Time values corresponding to the pitch.
    pitch_series (array-like): Quantized pitch values that need smoothing.
    smoothing_factor (float): Smoothing factor for the spline. Lower values = less smoothing.
    min_points (int): Minimum number of data points required to apply spline smoothing.

    Returns:
    smoothed_pitch (array-like): Smoothed pitch values over the entire time series.
    """
    # Convert input series to numpy arrays and handle None or NaN using pd.isna()
    time_series = np.array(time_series, dtype=float)
    pitch_series = np.array(pitch_series, dtype=float)

    # Initialize the result array with NaN values
    smoothed_pitch = np.full_like(pitch_series, np.nan)

    # Create a mask to filter out None/NaN values
    valid_mask = ~pd.isna(time_series) & ~pd.isna(pitch_series)

    # Find the indices of valid (non-NaN) data
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # If no valid data, return all NaNs
        return smoothed_pitch

    # Identify contiguous chunks of valid data
    contiguous_chunks = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)

    for chunk in contiguous_chunks:
        if len(chunk) >= min_points:  # Only process chunks with enough points
            time_chunk = time_series[chunk]
            pitch_chunk = pitch_series[chunk]

            # Normalize pitch_chunk to the range [0, 1]
            pitch_min = np.min(pitch_chunk)
            pitch_max = np.max(pitch_chunk)
            normalized_pitch_chunk = (pitch_chunk - pitch_min) / (pitch_max - pitch_min)

            # Apply smoothing to the normalized data
            spline_func = UnivariateSpline(time_chunk, normalized_pitch_chunk, s=smoothing_factor)
            smoothed_normalized_pitch = spline_func(time_chunk)

            # Rescale smoothed data back to the original range
            smoothed_pitch[chunk] = smoothed_normalized_pitch * (pitch_max - pitch_min) + pitch_min

        elif len(chunk) > 1:
            # If a chunk has fewer than min_points but more than 1, apply linear interpolation
            time_chunk = time_series[chunk]
            pitch_chunk = pitch_series[chunk]
            smoothed_pitch[chunk] = np.interp(time_chunk, time_chunk, pitch_chunk)

    return smoothed_pitch


def interpolate_below_length_old(arr, val, gap, indices=[]):
    """
    Interpolate gaps of value, <val> of
    length equal to or shorter than <gap> in <arr>,
    except for regions containing any index in <indices>.

    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number
    :param indices: List of indices where interpolation should not occur
    :type indices: list

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)

    # Identify regions to potentially interpolate
    if np.isnan(val):
        is_zero = np.isnan(s)
    else:
        is_zero = s == val

    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)

    # Loop through gaps to find ones that are below the gap length
    for i, d in enumerate(diff):
        if d <= gap:
            gap_start = int(i - d)
            gap_range = set(range(gap_start, i))

            # Check if any indices in the gap range are in the forbidden list
            if not gap_range.intersection(indices):
                s[gap_start:i] = np.nan  # Mark the region for interpolation

    # Perform interpolation only in marked regions
    interp = pd.Series(s).interpolate(method='linear', order=2, axis=0)\
                         .ffill()\
                         .bfill()\
                         .values

    return interp


def interpolate_below_length(arr, val, gap, indices=[]):
    """
    Interpolate gaps of value <val> in <arr>, but only if:
        - The gap length is <= <gap>
        - The gap does not intersect with any index in <indices>

    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate
    :type gap: number
    :param indices: List of indices where interpolation should not occur
    :type indices: list

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    indices = set(indices)

    if np.isnan(val):
        is_gap = np.isnan(s)
    else:
        is_gap = (s == val)

    # Identify all contiguous gaps
    in_gap = False
    gap_start = None
    gap_ranges = []

    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            gap_start = i
        elif not g and in_gap:
            in_gap = False
            gap_ranges.append((gap_start, i))

    if in_gap:
        gap_ranges.append((gap_start, len(s)))

    # Process gaps
    for start, end in gap_ranges:
        length = end - start
        if length > gap:
            continue  # Skip too-long gaps

        if any(idx in indices for idx in range(start, end)):
            continue  # Skip forbidden gaps

        # Mark for interpolation (set to NaN to trigger interpolation later)
        s[start:end] = np.nan

    # Perform interpolation only in marked regions (np.nan)
    series = pd.Series(s)
    interp = series.interpolate(method='linear') \
                    .ffill() \
                    .bfill() \
                    .values

    return interp


def align_time_series(*time_series):
    # Ensure at least one time series is provided
    if len(time_series) < 1:
        raise ValueError("At least one time series must be provided")

    # Compute the minimum timestep across all input time series
    min_timesteps = [np.min(np.diff(t[1])) for t in time_series]  # Find timestep for each series
    min_timestep = np.min(min_timesteps)  # Find the smallest timestep across all series

    # Initialize list to store the results for each time series
    interpolated_series = []

    # Loop over each time series
    for pitch, time in time_series:

        new_pitch, new_time = change_time_grid(pitch, time, min_timestep)

        # Append the new interpolated series (pitch, time) to the results
        interpolated_series.append((new_pitch, new_time))

    return interpolated_series, min_timestep


def change_time_grid(pitch, time, min_timestep):
    # Create a time grid for this series with the minimum timestep, preserving the original duration
    new_time = np.arange(time[0], time[-1] + min_timestep, min_timestep)

    # Interpolate the pitch data on the new time grid
    interp_func = interp1d(time, pitch, kind='linear', fill_value="extrapolate")
    new_pitch = interp_func(new_time)

    return new_pitch, new_time


def write_pkl(o, path):
    create_if_not_exists(path)
    with open(path, 'wb') as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    file = open(path,'rb')
    return pickle.load(file)


def remove_leading_trailing_nans(arr):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Find indices of non-NaN values
    mask = ~np.isnan(arr)

    # If all values are NaN, return an empty array
    if not mask.any():
        return np.array([])

    # Get the indices of the first and last non-NaN value
    first_valid = np.argmax(mask)
    last_valid = len(mask) - np.argmax(mask[::-1]) - 1

    # Slice the array to remove leading and trailing NaNs
    return arr[first_valid:last_valid+1]


def expand_zero_regions(arr, x):
    """
    Expands contiguous zero regions in a NumPy array by 'x' elements on both sides.
    The expansion is done by replacing the neighboring values with zeros.

    Parameters:
    arr (numpy array): Input 1D NumPy array.
    x (int): Number of elements to expand on either side of zero regions.

    Returns:
    numpy array: Array with expanded zero regions.
    """
    arr = arr.copy()

    # Identify the indices where zeros are present
    zero_mask = arr == 0

    # Use convolution to identify contiguous regions and their expansion
    expanded_mask = np.convolve(zero_mask.astype(int), np.ones(2 * x + 1, dtype=int), mode='same') > 0

    # Replace the expanded regions with 0s
    arr[expanded_mask] = 0

    return arr


def write_list_to_file(filename, my_list):
    # Open the file in write mode ('w')
    with open(filename, 'w') as file:
        # Iterate over each item in the list
        for item in my_list:
            # Write each item followed by a newline
            file.write(f"{item}\n")


def get_context(context_data, track, annot_ix, k, direction):

    cont = context_data[track][annot_ix]
    prec = cont['prec']
    succ = cont['succ']
    precsucc = [' '.join(x) for x in list(zip(prec, succ))]

    if direction == 'prec':
        if k > len(prec):
            return None
        else:
            return ' '.join(prec[:k])

    if direction == 'succ':
        if k > len(succ):
            return None
        else:
            return ' '.join(succ[:k])

    if direction == 'both':
        if k > len(precsucc):
            return None
        else:
            return ' '.join(precsucc[:k])

    raise Exception('direction must be prec, succ or both')


def append_row(df, row):
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    return df


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def append_to_csv(filepath, rows, overwrite):
    """
    Appends multiple rows to a CSV file or overwrites it if specified.

    :param filepath: Path to the CSV file.
    :param rows: A list of tuples, each representing a row of data.
    :param overwrite: If True, clears the file before writing. Default is False.
    """
    create_if_not_exists(filepath)
    mode = 'w' if overwrite else 'a'  # 'w' for overwrite, 'a' for append
    with open(filepath, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def format_time(seconds: float) -> str:
    """
    Converts a time in seconds to a formatted string "mm:ss:ms".

    :param seconds: Time in seconds.
    :return: Formatted time string.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1_000)
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"


def myround(x, base=100, method='round'):
    """
    Rounds x to the nearest multiple of base using numpy.

    :param x: Input number or array.
    :param base: The base to round to.
    :param method: 'floor' to round down, 'ceil' to round up.
    :return: Rounded number or array.
    """
    if method == 'floor':
        return base * np.floor(x / base)
    elif method == 'ceil':
        return base * np.ceil(x / base)
    elif method == 'round':
        return base * np.round(x / base)
    else:
        raise ValueError("method must be 'floor' or 'ceil'")


def group_contiguous_indices(indices):
    if not indices:
        return []

    indices = sorted(set(indices))  # Ensure sorted order and uniqueness
    grouped = []
    start = indices[0]

    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            grouped.append((start, indices[i - 1]))
            start = indices[i]

    grouped.append((start, indices[-1]))  # Add the last range
    return grouped

def shift_change_points(segment, num_shifted=5):
    """
    Augments time series by shifting change points in x and y directions.

    :param segment: Time series.
    :param num_shifted: Number of duplicate time series to generate.
    :return shifted: List of generated duplicate time series.
    """
    shifted = []

    # x axis to spline interpolate
    x = np.arange(len(segment))

    # Change points
    local_maxima = argrelextrema(segment, np.greater)[0]
    local_minima = argrelextrema(segment, np.less)[0]

    change_points = np.concatenate([local_maxima, local_minima])
    # Also add boundaries
    change_points = np.unique(np.concatenate([change_points, [0, len(segment) - 1]]))
    change_points = np.sort(change_points)

    # Add midpoints of changepoints to improve similarity
    mids = (change_points[1:] + change_points[:-1]) // 2
    change_points = np.unique(np.concatenate([change_points, mids]))
    # Do it again
    mids = (change_points[1:] + change_points[:-1]) // 2
    change_points = np.unique(np.concatenate([change_points, mids]))

    for i in range(num_shifted):
        # Shift the change points
        noise = np.floor(np.random.rand(len(change_points) - 2) * 4 - 2)
        change_points[1:-1] = change_points.astype(np.float64)[1:-1] + noise
        change_points = np.unique(change_points)

        # Get shifted magnitudes and add noise
        change_points_value = np.interp(change_points, x, segment) + np.floor(np.random.rand(len(change_points)) * 5 - 2.5)

        # Interpolate to get smooth curve
        spline = make_interp_spline(change_points, change_points_value)
        y = spline(x)

        # Resample and smooth to get different lengths
        y = resample(y, len(segment) + np.random.randint(len(segment) // 5) - (len(segment) // 10))
        y = savgol_filter(y, 15, 5)

        shifted.append(y)

    return shifted
