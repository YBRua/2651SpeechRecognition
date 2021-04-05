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
    short_time_zcr = []
    for arr in data:
        signs = np.sign(arr)
        diff = np.diff(signs)
        zcr = np.sum(np.abs(diff)) / (2*len(arr))
        short_time_zcr.append(zcr)

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
    short_time_energy = []
    for arr in data:
        energy = np.sum(np.multiply(arr, window)**2)
        if take_sqrt:
            energy = np.sqrt(energy)
        short_time_energy.append(energy)

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
    short_time_magnitude = []
    for arr in data:
        magnitude = np.sum(np.multiply(np.abs(arr), window))
        short_time_magnitude.append(magnitude)

    return short_time_magnitude


def extract_short_time_features(
        data, use_window='hamming',
        frame_size=512,
        frame_shift=128):
    """Divide .wav data into frames and extract short-time features.

    Arguments:
        data: list -- wav data read from a .wav file
        window: list -- window to be applied on each frame (default: hamming)
        frame_size: int -- num of samples in a frame (default: 512)
        frame_shift: int -- num of samples skipped per shift (default: 128)

    Returns:
        mag: list -- short-time magnitude
        eng: list -- short-time (square-rooted) energy
        zcr: list -- short-time zero-crossing rate
    """
    window = signal.get_window(use_window, frame_size)
    frames = segmentation(data, frame_size, frame_shift)
    mag = signal.medfilt(magnitude(frames, window), 3)
    eng = signal.medfilt(energy(frames, window), 3)
    zcr = signal.medfilt(zero_crossing_rate(frames), 3)

    return mag, eng, zcr
