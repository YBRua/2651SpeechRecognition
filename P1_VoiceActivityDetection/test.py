# %%
import typing
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from vad_utils import parse_vad_label


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
        take_log: boolean -- if True, returns the log energy (Default: True)

    Returns:
        short_time_energy -- a list of energy
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
    short_time_magnitude = []
    for arr in data:
        magnitude = np.sum(np.multiply(np.abs(arr), window))
        short_time_magnitude.append(magnitude)

    return short_time_magnitude


FRAME_SIZE = 0.025   # 25ms per frame
FRAME_SHIFT = 0.010  # 10ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

path = './wavs/dev/54-121080-0043.wav'
# path = './data_so.wav'
rate, data = wavfile.read(path)

frames = segmentation(data, N_FRAME, N_SHIFT)
window = signal.windows.hamming(N_FRAME)
mag = magnitude(frames, window)
enr = energy(frames, window)
zcr = zero_crossing_rate(frames)
zcr = signal.medfilt(zcr, 5)

label = parse_vad_label('0.22,1.71 3.07,3.84 3.85,4.24 4.73,6.24 6.41,6.83 6.93,7.6 7.77,10.32 10.8,12.55 13.05,14.01', FRAME_SIZE, FRAME_SHIFT)
padded_label = np.pad(label, (0, np.maximum(len(frames) - len(label), 0)))[:len(frames)]

fig, axs = plt.subplots(3, 1, figsize=(25,10))
axs[0].plot(data)
axs[0].set_title('Voice Signal')
axs[1].plot(zcr)
axs[1].set_title('Short-time ZCR')
axs[2].plot(enr)
axs[2].set_title('Short-time Energy')
axs[1].scatter(range(len(padded_label)), padded_label, s=0.4, c='r')

# %%

f, t, zxx = signal.stft(data, rate, window='hamming', nperseg=400, noverlap=240)
fg, ax = plt.subplots(2, 1, figsize=(25, 10),gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax[0].imshow(np.abs(zxx), cmap='plasma')
ax[1].scatter(range(len(padded_label)), padded_label, s=0.4, c='r')
ax[1].set_xlim(0,)
