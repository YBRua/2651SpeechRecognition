# %%
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from vad_utils import parse_vad_label
from short_time_features import segmentation
from short_time_features import magnitude, energy, zero_crossing_rate
from short_time_features import binned_stft

FRAME_SIZE = 0.032   # 25ms per frame
FRAME_SHIFT = 0.008  # 10ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

path = './wavs/dev/472-130755-0013.wav'
# path = './data_so.wav'
rate, data = wavfile.read(path)
data = np.array(data - np.mean(data))

# %%
frames = segmentation(data, N_FRAME, N_SHIFT)
window = signal.windows.hamming(N_FRAME)
mag = magnitude(frames, window)
enr = energy(frames, window)
zcr = zero_crossing_rate(frames)
zcr = signal.medfilt(zcr, 5)
enr = signal.medfilt(enr, 9)
mag = signal.medfilt(mag, 3)

labels = parse_vad_label('0.15,2.03 2.85,5.54 6.09,7.47 7.52,10.49 10.5,11.93 13.73,14.63 14.7,15.32')
labels = np.pad(labels, (0, np.maximum(len(frames) - len(labels), 0)))[:len(frames)]

fig, axs = plt.subplots(3, 1, figsize=(25, 10))
axs[0].plot(data)
axs[0].set_title('Voice Signal')
axs[1].plot(zcr)
axs[1].set_title('Short-time ZCR')
axs[2].plot(mag)
axs[2].set_title('Short-time Energy')

# %%

f, t, zxx = signal.stft(data, rate, window='hamming', nperseg=N_FRAME, noverlap=N_FRAME-N_SHIFT)
fg, ax = plt.subplots(2,1, figsize=(25, 10), gridspec_kw={'height_ratios':[3,1]})
im = ax[0].imshow(20 * np.log10(np.abs(zxx)), cmap='plasma')
fg.colorbar(im, ax=ax[0], orientation='horizontal')
ax[1].scatter(range(len(labels)), labels, c='#ff0000')
ax[1].set_xlim(0, zxx.shape[1])

# %%
bins = binned_stft(data)
for b in bins:
    val = np.mean(np.where(labels == 1, b, 0))
    print(val)
    val = np.mean(np.where(labels == 0, b, 0))
    print(val)
    print()
