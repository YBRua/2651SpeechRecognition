import librosa
import numpy as np


def extract_mfcc(mel_s, rate, n_mfcc=20):
    """Extracts MFCC features.
    Includes MFCC and 1st- and 2nd-order deltas.

    Arguments:
        mel_s: 2darray -- Mel-Field Spectrogram.
        rate: int -- Sample rate of input.
        n_mfcc: int -- n_mfcc passed into librosa.feature.mfcc()

    Returns:
        feature: 1darray -- MFCC and deltas, concatenated into an array.
    """
    mfcc = librosa.feature.mfcc(sr=rate, S=mel_s, n_mfcc=n_mfcc)
    mfcc_d1 = librosa.feature.delta(mfcc, order=1)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    feature = np.concatenate([mfcc, mfcc_d1, mfcc_d2], axis=0)

    return feature


def rms_energy(stft_s):
    """Computes the rms energy of each frame.

    Arguments:
        stft_s: 2darray -- STFT spectrogram.
        rate: int -- Sample rate of input.

    Returns:
        rms: 1darray -- rms energy.
    """
    mag, _ = librosa.core.magphase(stft_s)
    rms = librosa.feature.rms(S=mag)

    return rms


def spectral_feature_extractor(
    data, rate=16000,
    n_frame=512, n_shift=128,
    use_window='hann'
):
    stft = librosa.core.stft(
        data,
        hop_length=n_shift, win_length=n_frame,
        window=use_window)
    mel_s = librosa.feature.melspectrogram(sr=rate, S=np.abs(stft)**2)

    mfcc = extract_mfcc(mel_s, rate)
    rms = rms_energy(stft)

    return np.concatenate([mfcc, rms], axis=0)
