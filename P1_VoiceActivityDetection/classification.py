# %% libiraries and class and func defs
import os
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from short_time_features import feature_extraction
from vad_utils import pad_labels
from vad_utils import read_label_from_file
from evaluate import get_metrics


# always run short_time_analysis.py
# before calling the functions below
def load_all_data(
    data_set_path,
    label_path,
    use_window='hamming',
    frame_size=512,
    frame_shift=128,
    medfilt_size=3,
    bin_mode='coarse',
):
    """Load and convert all data into a single array of frames and labels.
    Note that the arguments should be EXACTLY THE SAME AS
    the aruguments used in feature analysis.

    Returns:
        all_frames -- Array of [N,6], rows are frames and columns are features
        all_labels -- Array of N where each element is a label of a frame
    """
    labels = read_label_from_file(label_path)
    all_frames = np.zeros([0, 6])
    all_labels = np.zeros([0])

    for root, dirs, files in os.walk(data_set_path):
        for index, f in enumerate(tqdm(files)):
            if '.wav' in f:
                rate, raw_data = wavfile.read(os.path.join(data_set_path, f))
                data = np.array(raw_data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalize

                frames = feature_extraction(
                    data,
                    use_window,
                    frame_size,
                    frame_shift,
                    medfilt_size,
                    bin_mode,
                    rate
                    ).T
                ground_truth = pad_labels(
                    labels[f.split('.wav')[0]], frames.shape[0])
                all_frames = np.concatenate([all_frames, frames], axis=0)
                all_labels = np.concatenate([all_labels, ground_truth], axis=0)

    return all_frames, all_labels


def quick_pass(classifier, frames, labels):
    """If the data and labels are ready
    this function can be called to evaluate the model directly

    Returns:
        auc, err -- metrics of current classifier on the current dataset
    """
    pred = classifier.pred(frames)
    auc, eer = get_metrics(pred, labels)
    return auc, eer
