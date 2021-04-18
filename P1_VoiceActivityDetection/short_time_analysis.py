# %% libraries and function defs
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from vad_utils import read_label_from_file, pad_labels
from short_time_features import extract_time_domain_features
from short_time_features import binned_stft


def time_domain_features(
        index, file_name, data, label, df,
        frame_size=512, frame_shift=128,
        use_window='hamming', medfilt_size=3):
    mag, eng, zcr = extract_time_domain_features(
        data, use_window,
        frame_size, frame_shift, medfilt_size)

    voice_mag = np.mean(np.where(label == 1, mag, 0))
    unvoice_mag = np.mean(np.where(label == 0, mag, 0))
    voice_eng = np.mean(np.where(label == 1, eng, 0))
    unvoice_eng = np.mean(np.where(label == 0, eng, 0))
    voice_zcr = np.mean(np.where(label == 1, zcr, 0))
    unvoice_zcr = np.mean(np.where(label == 0, zcr, 0))

    current = df.append(pd.DataFrame([[
        file_name,
        voice_mag, unvoice_mag,
        voice_eng, unvoice_eng,
        voice_zcr, unvoice_zcr]], index=[index], columns=df.columns))

    return current


def freq_domain_features(
        index, file_name, data, label, df,
        frame_size=512, frame_shift=128,
        use_window='hamming', sample_rate=16000, bin_mode='coarse'):
    if bin_mode != 'coarse':
        raise ValueError('Binning mode other than coarse is not supported.')

    bins = binned_stft(
        data,
        use_window, bin_mode,
        frame_size, frame_shift, sample_rate)

    results = [file_name]
    for b in bins:
        results.append(np.mean(np.where(label == 1, b, 0)))
        results.append(np.mean(np.where(label == 0, b, 0)))

    current = df.append(
        pd.DataFrame([results], index=[index], columns=df.columns))

    return current


def feature_analysis(
        path, labels,
        frame_size=512, frame_shift=128,
        use_window='hamming', medfilt_size=3):
    time_columns = [
        'File',
        'Voiced Magnitude', 'Unvoiced Magnitude',
        'Voiced Energy', 'Unvoiced Energy',
        'Voiced ZCR', 'Unvoiced ZCR'
    ]
    freq_columns = [
        'File',
        'Voiced LowFreq', 'Unvoiced LowFreq',
        'Voiced MedFreq', 'Unvoiced MedFreq',
        'Voiced HighFreq', 'Unvoiced HighFreq'
    ]
    time_analysis = pd.DataFrame(columns=time_columns)
    freq_analysis = pd.DataFrame(columns=freq_columns)

    for root, dirs, files in os.walk(path):
        for index, f in enumerate(tqdm(files)):
            if '.wav' in f:
                rate, raw_data = wavfile.read(os.path.join(path, f))
                data = np.array(raw_data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalization

                length = int(np.ceil(
                    (len(data)-(frame_size-frame_shift)) / frame_shift))
                label = pad_labels(labels[f.split('.wav')[0]], length)

                time_analysis = time_domain_features(
                    index, f, data, label, time_analysis)
                freq_analysis = freq_domain_features(
                    index, f, data, label, freq_analysis)

    return time_analysis, freq_analysis


# %% short-time feature extraction
data_path = './wavs/dev'
labels = read_label_from_file()
time, freq = feature_analysis(data_path, labels)

time.to_csv('./time_domain_features.csv', index=False)
freq.to_csv('./freq_domain_features.csv', index=False)

# %% load files
time = pd.read_csv('./time_domain_features.csv')
freq = pd.read_csv('./freq_domain_features.csv')
