# %% libiraries and class and func defs
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from short_time_features import feature_extraction
from vad_utils import pad_labels
from vad_utils import read_label_from_file
from evaluate import get_metrics
from classifiers.basic import BasicThresholdClassifer


# always run short_time_analysis.py
# before calling the functions below
def load_all_data(
    dev_set_path='./wavs/dev',
    label_path='./data/dev_label.txt'
):
    labels = read_label_from_file(label_path)
    all_frames = np.zeros([0, 6])
    all_labels = np.zeros([0])

    for root, dirs, files in os.walk(dev_set_path):
        for index, f in enumerate(tqdm(files)):
            if '.wav' in f:
                rate, raw_data = wavfile.read(os.path.join(dev_set_path, f))
                data = np.array(raw_data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalize

                frames = feature_extraction(data).T
                ground_truth = pad_labels(
                    labels[f.split('.wav')[0]], frames.shape[0])
                all_frames = np.concatenate([all_frames, frames], axis=0)
                all_labels = np.concatenate([all_labels, ground_truth], axis=0)

    return all_frames, all_labels


def run_on_devset(
    classifier,
    dev_set_path='./wavs/dev',
    label_path='./data/dev_label.txt'
):
    """Run the classifier on the given trainning or developing set.

    Arguments:
    classifier -- A classifier for VAD tasks
    dev_set_path -- path to dev sets
    label_path -- path to unparsed labels (a .txt file)
                  (default: ./data/dev_label.txt)

    Returns:
    auc, eer -- metrics of current classifier on current data set.
    """
    frames, labels = load_all_data(dev_set_path, label_path)
    pred = classifier.pred(frames)

    auc, eer = get_metrics(pred, labels)
    print('Run Finished.')
    print('  - Average AUC: {:.4f}'.format(auc))
    print('  - Average EER: {:.4f}'.format(eer))

    return auc, eer


def quick_pass(classifier, frames, labels):
    pred = classifier.pred(frames)
    auc, eer = get_metrics(pred, labels)
    return auc, eer


if __name__ == '__main__':
    time = pd.read_csv('./time_domain_features.csv')
    freq = pd.read_csv('./freq_domain_features.csv')
    data_path = ('./wavs/dev')
    classifier = BasicThresholdClassifer(time, freq)
    run_on_devset(classifier, data_path)
