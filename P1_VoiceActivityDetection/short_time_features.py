# %%
import typing
import numpy as np
import scipy.signal as signal


def segmentation(
        arr,
        frame_size,
        frame_shift) -> typing.List[typing.List[int]]:
    """Divides input array into frames.
    Pad zeros to the last list if it is not long enough.

    Arguments:
        arr: list -- input array to be divided
        frame_size: int -- n samples per frame
        frame_shift: int -- n samples for frame shift

    Returns:
        frames -- a list of segmented samples
    """
    arr = list(arr)
    frames = []
    for start in range(0, len(arr) - frame_size, frame_shift):
        frames.append(arr[start: start+frame_size])
    start += frame_size
    while(len(arr[start:])) < frame_size:
        arr.append(0)
    frames.append(arr[start:])

    return np.array(frames)


def zero_crossing_rate(data) -> typing.List[float]:
    """Computes the short-time zero-crossing rate.

    Arguments:
        data: 2d-array -- framed voice data

    Returns:
        short_time_zcr -- a list of zero-crossing rate
    """
    signs = np.sign(data)
    diff = np.diff(signs)
    short_time_zcr = np.sum(np.abs(diff), axis=-1) / (2*len(data[0]))

    return short_time_zcr


def energy(data, window, take_sqrt=True) -> typing.List[float]:
    """Computes the short-time energy.

    Arguments:
        data: 2d-array -- framed voice data
        window: array -- window for the frame.
        take_sqrt: boolean -- if True, returns the square root of energy
                                (Default: True)

    Returns:
        short_time_energy -- a list of short time energy
    """
    short_time_energy = np.sum(np.multiply(data, window)**2, axis=-1)
    if take_sqrt:
        short_time_energy = np.sqrt(short_time_energy)

    return short_time_energy


def log_energy(data, window) -> typing.List[float]:
    en = np.array(energy(data, window, take_sqrt=False))
    return np.log(en)


def magnitude(data, window) -> typing.List[float]:
    """Computes the short time magnitude.

    Arguments:
        data: 2d-array -- framed voice data
        window: array -- window for each frame.

    Returns:
        short_time_magnitude -- a list of short time magnitude
                                computed at each frame.
    """
    short_time_magnitude = np.sum(np.multiply(np.abs(data), window), axis=-1)

    return short_time_magnitude


def extract_short_time_features(
        data,
        use_window='hamming',
        frame_size=512,
        frame_shift=128,
        medfilt_size=3):
    """Divide .wav data into frames and extract short-time features.

    NOTE: this function does not remove dc-offset
    nor does it normalize the input
    so remember to do them manually

    Arguments:
        data: 1darray -- wav data read from a .wav file
        use_window: string -- window to be applied on each frame
                              (default: hamming)
        frame_size: int -- num of samples in a frame (default: 512)
        frame_shift: int -- num of samples skipped per shift (default: 128)
        medfilt_size: int -- size of median filter, no filtering if 0
                             (default: 3)

    Returns:
        mag: list -- short-time magnitude
        eng: list -- short-time (square-rooted) energy
        zcr: list -- short-time zero-crossing rate
    """
    window = signal.get_window(use_window, frame_size)
    frames = segmentation(data, frame_size, frame_shift)
    mag = magnitude(frames, window)
    eng = energy(frames, window)
    zcr = zero_crossing_rate(frames)

    if medfilt_size != 0:
        mag = signal.medfilt(mag, medfilt_size)
        eng = signal.medfilt(eng, medfilt_size)
        zcr = signal.medfilt(zcr, medfilt_size)

    return mag, eng, zcr


def binned_stft(
        data,
        use_window='hamming',
        bin_mode='coarse',
        frame_size=512,
        frame_shift=128,
        sample_rate=16000):
    """Performs STFT on input signal and bin signal according to frequencies.

    Arguments:
        data: 1darray -- wav data read from a .wav file
        use_window: string -- window to be applied on each frame
                              (default: hamming)
        bin_mode: string -- binning mode.
                            coarse: divide frequencies into 3 bins,
                                boundary: 500Hz and 3000Hz
                            fine: divied frequencies into fine-grained bins,
                                boundary: 32Hz, 64Hz, ..., 2**iHz
                            (default: 'coarse')
        frame_size: int -- num of samples in a frame
                           (default: 512)
        frame_shift: int -- num of samples skipped per shift
                            (default: 128)

    Returns:
        binned_energy: list -- binned rms energy in frequency domain
    """
    overlap_size = frame_size - frame_shift
    freq, time, zxx = signal.stft(
        data,
        sample_rate,
        use_window,
        frame_size,
        overlap_size,
        boundary=None)
    freq = np.expand_dims(freq, -1)
    binned_energy = []
    if bin_mode == 'fine':
        i = 5
        while 2 ** i < sample_rate / 2:
            lower_bound = 2 ** i - 1
            upper_bound = 2 ** (i+1) - 1
            valid = np.where((lower_bound <= freq) & (freq < upper_bound), zxx, 0)
            rms = np.sqrt(np.mean(np.abs(valid)**2, axis=0))
            binned_energy.append(rms)
            i += 1
    elif bin_mode == 'coarse':
        div = [500, 3000, sample_rate]
        lower_bound = 0
        for upper_bound in div:
            valid = np.where((lower_bound <= freq) & (freq < upper_bound), zxx, 0)
            rms = np.sqrt(np.mean(np.abs(valid)**2, axis=0))
            binned_energy.append(rms)
            lower_bound = upper_bound
    else:
        raise ValueError(
            "Binning mode {:s} is not supported.".format(bin_mode),
            "Valid params are: ['coarse', 'fine']")

    return binned_energy
