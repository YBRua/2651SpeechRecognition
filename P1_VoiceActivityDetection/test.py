# %%
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from vad_utils import parse_vad_label
from short_time_features import segmentation
from short_time_features import magnitude, energy, zero_crossing_rate
from short_time_features import binned_stft

FRAME_SIZE = 0.032 * 2   # 100ms per frame
FRAME_SHIFT = 0.008 * 2  # 40ms frame-shift
SAMPLE_RATE = 16000      # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

path = './wavs/dev/1031-133220-0062.wav'
# path = './data_so.wav'
rate, data = wavfile.read(path)
data = np.array(data - np.mean(data))

# %%
frames = segmentation(data, N_FRAME, N_SHIFT)
window = signal.windows.hamming(N_FRAME)
mag = magnitude(frames, window)
enr = energy(frames, window)
zcr = zero_crossing_rate(frames)
zcr = signal.medfilt(zcr, 3)
enr = signal.medfilt(enr, 9)
mag = signal.medfilt(mag, 3)

labels = parse_vad_label('0.44,2.02 2.05,5.67 6.14,10.05 10.47,11.58 11.67,12.59 13.43,15.13')
labels = np.pad(labels, (0, np.maximum(len(frames) - len(labels), 0)))[:len(frames)]

fig, axs = plt.subplots(3, 1, figsize=(25, 10))
axs[0].plot(data)
axs[0].set_title('Voice Signal')
axs[1].plot(zcr)
axs[1].set_title('Short-time ZCR')
axs[1].scatter(range(len(labels)), labels, c='#ff0000')
axs[2].plot(mag)
axs[2].set_title('Short-time Energy')
