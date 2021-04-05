# TODO: extract short-time features
#       and do some statistics.

# %%
import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from vad_utils import parse_vad_label, read_label_from_file
from short_time_features import extract_short_time_features


FRAME_SIZE = 0.032
FRAME_SHIFT = 0.008
SAMPLE_RATE = 16000
N_FRAME = (SAMPLE_RATE * FRAME_SIZE)
N_SHIFT = (SAMPLE_RATE * FRAME_SHIFT)

path = './wavs/dev'
window = signal.windows.hamming(N_FRAME)
vad_dict = read_label_from_file()

for root, dirs, files in os.walk(path):
    for wavf in files:
        if '.wav' in wavf:
            _, data = wavfile.read(os.path.join(root, wavf))
            mag, eng, zcr = extract_short_time_features(data)
