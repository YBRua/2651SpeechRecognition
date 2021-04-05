# %%
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from vad_utils import parse_vad_label
from short_time_features import segmentation
from short_time_features import magnitude, energy, zero_crossing_rate


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
enr = signal.medfilt(enr, 3)
mag = signal.medfilt(mag, 3)

label = parse_vad_label('0.22,1.71 3.07,3.84 3.85,4.24 4.73,6.24 6.41,6.83 6.93,7.6 7.77,10.32 10.8,12.55 13.05,14.01', FRAME_SIZE, FRAME_SHIFT)
padded_label = np.pad(label, (0, np.maximum(len(frames) - len(label), 0)))[:len(frames)]

fig, axs = plt.subplots(3, 1, figsize=(25, 10))
axs[0].plot(data)
axs[0].set_title('Voice Signal')
axs[1].plot(zcr)
axs[1].set_title('Short-time ZCR')
axs[2].plot(enr)
axs[2].set_title('Short-time Energy')
axs[1].scatter(range(len(padded_label)), padded_label, s=0.4, c='r')

# %%

f, t, zxx = signal.stft(data, rate, window='hamming', nperseg=400, noverlap=240)
fg, ax = plt.subplots(2, 1, figsize=(25, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax[0].imshow(np.abs(zxx), cmap='plasma')
ax[1].scatter(range(len(padded_label)), padded_label, s=0.4, c='r')
ax[1].set_xlim(0,)
