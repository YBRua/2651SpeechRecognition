# %% libiraries and class and func defs
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm

from short_time_features import feature_extraction
from vad_utils import pad_labels
from vad_utils import read_label_from_file
from evaluate import get_metrics
from classifiers.basic import BasicThresholdClassifer

# %% load files
# always run short_time_analysis.py
# before running the lines below
time = pd.read_csv('./time_domain_features.csv')
freq = pd.read_csv('./freq_domain_features.csv')

# %% validate
data_path = './wavs/dev'
labels = read_label_from_file()
classifier = BasicThresholdClassifer(time, freq)


auc_list = []
eer_list = []
for root, dirs, files in os.walk(data_path):
    for index, f in enumerate(tqdm(files)):
        if '.wav' in f:
            pred = []
            rate, raw_data = wavfile.read(os.path.join(data_path, f))
            data = np.array(raw_data, dtype=float)
            data -= np.mean(data)   # remove dc-offset
            data /= 32767           # normalization

            results = feature_extraction(data).T
            for frame in results:
                p = classifier.pred(frame)
                pred.append(p)

            ground_truth = pad_labels(labels[f.split('.wav')[0]], len(pred))
            auc, eer = get_metrics(pred, ground_truth)
            auc_list.append(auc)
            eer_list.append(eer)

# %%
print('Validation Finished.')
print('  - Average AUC: {:.2f}'.format(np.mean(auc_list)*100))
print('  - Average EER: {:.2f}'.format(np.mean(eer_list)))
plt.scatter(range(len(auc_list)), auc_list)
