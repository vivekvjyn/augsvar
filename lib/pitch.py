import math
import copy
import numpy as np
from scipy.signal import find_peaks

from transwara.tools.fcc import getChainCode, n_coords
from transwara.tracks import get_stability_mask
from transwara.utils import (
    smooth_pitch_curve, expand_zero_regions,
    interpolate_below_length, pitch_seq_to_cents,
    subsample_series)

def process_pitch(pitch, time, timestep, interp, tonic, beat_length, subsample, smoothing_factor, boundaries_seq=[]):

    pitch = expand_zero_regions(pitch, round(0.02/timestep))

    # Interpolation
    pitch = interpolate_below_length(pitch, 0, (interp*0.001/timestep), boundaries_seq)

    null_ind = pitch==0

    pitch[pitch<50]=0
    pitch[null_ind]=0

    # cents
    pitch = pitch_seq_to_cents(pitch, tonic=tonic)

    # subsample
    time, pitch = subsample_series(time, pitch, subsample)

    # beats
    time_beat = time/beat_length

    # smoothing
    pitch = smooth_pitch_curve(time_beat, pitch, smoothing_factor=smoothing_factor)

    return pitch, time_beat


def mask_stability(pitch, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):

    stab_mask = get_stability_mask(
                    pitch, min_stability_length_secs=min_stability_length_secs,
                    stability_hop_secs=stability_hop_secs, var_thresh=var_thresh,
                    timestep=timestep)

    stable_ix = np.where(stab_mask==1)[0]
    stable_ix = [x for x in stable_ix if pitch[x] != None and not np.isnan(pitch[x])]

    pitch = copy.deepcopy(pitch)
    pitch[stable_ix] = np.nan

    pitch = np.where(np.isnan(pitch), None, pitch).astype(object)

    return pitch, stable_ix



def get_prop_octave(pitch, o=1):
    madyhama = len(np.where(np.logical_and(pitch>=0, pitch<=1200))[0]) # middle octave
    tara = len(np.where(pitch>1200)[0]) # higher octave
    mandhira = len(np.where(pitch<0)[0]) # lower octave

    octs = [mandhira, madyhama, tara]
    return octs[o]/len(pitch)


def transpose_pitch(pitch):
    ## WARNING: Assumes no pitch values in middle octave+2 or middle octave-2
    ## and that no svara traverses two octaves (likely due to pitch errors)
    r_prop = get_prop_octave(pitch, 0)
    p_prop = get_prop_octave(pitch, 1)
    i_prop = get_prop_octave(pitch, 2)

    if r_prop == 0 and i_prop == 0:
        # no transposition
        return pitch, False

    if r_prop == 0 and p_prop == 0:
        # transpose down
        return pitch-1200, True

    if i_prop == 0 and p_prop == 0:
        # transpose up
        return pitch+1200, True

    if i_prop > 0.6:
        # transpose down
        return pitch-1200, True

    return pitch, False


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



def get_samps(pitch_cents, i1, i2, prec_suc=80):
    psamp = pitch_cents[i1:i2]

    l = len(pitch_cents)
    ps1 = max([0,i1-prec_suc])
    ps2 = min([l-1,i2+prec_suc])

    presamp = pitch_cents[ps1:i1]
    possamp = pitch_cents[i2:ps2]

    return psamp, presamp, possamp


def trim_curve_ix(pitch_curve):

    m = pitch_curve != None

    if not False in m:
        return pitch_curve, 0, 0

    (led,tral) = m.argmax()-1, m.size - m[::-1].argmax()

    trimmed_curve = pitch_curve[led+1:tral]

    return trimmed_curve, led+1, tral


def segment_nones(pitch):

    segments_ix = []
    this_segment = []
    start = True
    for i,p in enumerate(pitch):
        if start == True and p:
            this_segment.append(i)
            start = False
        if start == False and not p:
            this_segment.append(i)
            segments_ix.append(this_segment)
            start = True
            this_segment = []

    return segments_ix



# get change points
def get_change_points(samp, prom, height=None):

    peaks = find_peaks(samp, prominence=prom, height=height)[0]
    troughs = find_peaks(-samp, prominence=prom, height=height)[0]

    pat = np.concatenate([peaks, troughs])

    return list(sorted(set(pat)))


def curvature(x, y):
    # Calculate the first derivatives
    if len(x)<2:
        return []
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Calculate the second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Compute the curvature using the formula
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    return curvature


#def find_elbows_knees(data, prominence=0.03):
#    x = np.arange(len(data))
#    y = data

#    # Calculate curvature
#    curv = curvature(x, y)

#    # Find peaks in the curvature which correspond to elbows/knees
#    peaks, _ = find_peaks(curv, prominence=prominence)
#
#    return list(sorted(set(peaks)))

def find_elbows_knees(data, prominence=0.04):
    # Prepare to collect results
    all_peaks = []

    # Initialize variables for tracking the start of each chunk
    start_index = 0

    while start_index < len(data):
        # Find the first non-None value
        while start_index < len(data) and data[start_index] is None:
            start_index += 1

        if start_index >= len(data):
            break

        # Find the next None value (end of the current chunk)
        end_index = start_index
        while end_index < len(data) and data[end_index] is not None:
            end_index += 1

        # Extract the current chunk
        chunk = data[start_index:end_index]

        # Calculate indices for the chunk
        x = np.arange(len(chunk))
        y = np.array(chunk, dtype=float)

        # Calculate curvature
        curv = curvature(x, y)

        if not len(curv)==0:
            # Find peaks in the curvature which correspond to elbows/knees
            peaks, _ = find_peaks(curv, prominence=prominence)

            # Adjust the peaks to the original indices
            adjusted_peaks = peaks + start_index

            # Append the adjusted peaks to the result
            all_peaks.extend(adjusted_peaks)

        # Move to the next chunk
        start_index = end_index

    return sorted(set(all_peaks))


# break down into sections
def encode_sample(psamp, seg_points):
    seg_points = list(sorted(set([0] + seg_points + [len(psamp)])))
    chunks = [psamp[seg_points[i-1]:seg_points[i]] for i in range(1,len(seg_points))]

    if len(chunks) == 0:
        import ipdb; ipdb.set_trace()

    n_cat = int(n_coords/2)+1

    mins = []
    maxs = []
    durations = []
    fccs = []
    averages = []
    stds = []
    for c in chunks:
        mins.append(np.min(c))
        maxs.append(np.max(c))
        #direcs.append(c[-1] > c[0])
        durations.append(len(c))
        averages.append(np.mean(c))
        stds.append(np.std(c))
        #pitch75.append(np.quantile(c, 0.75))
        #pitch25.append(np.quantile(c, 0.25))

        x0 = 0
        y0 = c[0]
        x1 = len(c)*10
        y1 = c[-1]

        fcc = getChainCode(x0,y0,x1,y1)
        onehot = np.zeros(n_cat)
        onehot[fcc] = 1
        fccs.append(list(onehot))

    encoded = []
    for i in range(len(chunks)):
        encoded.append(fccs[i] + [durations[i]] + [mins[i]] + [maxs[i]] + [averages[i]] + [stds[i]])

    return encoded


def get_contextual_features(presamp, psamp, possamp, cp):

    psamp = psamp.copy().astype(float)
    presamp = presamp.copy().astype(float)
    possamp = possamp.copy().astype(float)

    psamp[np.where(psamp==None)[0]] = np.nan
    presamp[np.where(presamp==None)[0]] = np.nan
    possamp[np.where(possamp==None)[0]] = np.nan

    #'pitch_range'
    #'av_pitch'
    #'min_pitch'
    #'max_pitch'
    #'pitch75'
    #'pitch25'
    #'av_first_pitch'
    #'av_end_pitch'
    #'num_change_points_pitch'
    #'prec_pitch_range'
    #'av_prec_pitch'
    #'min_prec_pitch'
    #'max_prec_pitch'
    #'prec_pitch75'
    #'prec_pitch25'
    #'succ_pitch_range'
    #'av_succ_pitch'
    #'min_succ_pitch'
    #'max_succ_pitch'
    #'succ_pitch75'
    #'succ_pitch25'
    #'direction_asc'
    #'direction_desc'

    features = [
        np.nanmax(psamp)-np.nanmin(psamp),
        np.nanmean(psamp),
        np.nanmin(psamp),
        np.nanmax(psamp),
        np.quantile(psamp, 0.75),
        np.quantile(psamp, 0.25),
        np.nanmean(psamp[:int(len(psamp)*0.1)]),
        np.nanmean(psamp[-int(len(psamp)*0.1):]),
        len(cp),
        np.nanmax(presamp)-np.nanmin(presamp) if len(presamp)!=0 else np.nan,
        np.nanmean(presamp) if len(presamp)!=0 else np.nan,
        np.nanmin(presamp) if len(presamp)!=0 else np.nan,
        np.nanmax(presamp) if len(presamp)!=0 else np.nan,
        np.quantile(presamp, 0.75) if len(presamp)!=0 else np.nan,
        np.quantile(presamp, 0.25) if len(presamp)!=0 else np.nan,
        np.nanmax(possamp)-np.nanmin(possamp) if len(possamp) != 0 else np.nan,
        np.nanmean(possamp) if len(possamp) != 0 else np.nan,
        np.nanmin(possamp) if len(possamp) != 0 else np.nan,
        np.nanmax(possamp) if len(possamp) != 0 else np.nan,
        np.quantile(possamp, 0.75) if len(possamp) != 0 else np.nan,
        np.quantile(possamp, 0.25) if len(possamp) != 0 else np.nan,
        int(np.nanmean(psamp[:int(len(psamp)*0.1)]) < np.nanmean(psamp[-int(len(psamp)*0.1):])),
        int(np.nanmean(psamp[:int(len(psamp)*0.1)]) > np.nanmean(psamp[-int(len(psamp)*0.1):]))
    ]

    return features


def trailing_vals(arr, reverse=False, val=None):
    if reverse:
        arr = arr[::-1]

    new_arr = []
    for x in arr:
        if x != val:
            new_arr.append(x)
        else:
            break
    new_arr = np.array(new_arr)
    if reverse:
        return new_arr[::-1]
    else:
        return new_arr
