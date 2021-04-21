# %% import and initialize
import os
import pickle
import pandas as pd
from tqdm import trange

from vad_utils import read_label_from_file
from classifiers.basic import BasicThresholdClassifer
from classification import load_all_data, quick_pass


max_iter = 5000
time = pd.read_csv('./time_domain_features.csv')
freq = pd.read_csv('./freq_domain_features.csv')
data_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'
labels = read_label_from_file()

# baseline: default params
# auc: 0.9051
# eer: 0.1508
classifier = BasicThresholdClassifer(time, freq)
baseline_auc = 0.9051
baseline_eer = 0.1508
highest = baseline_auc
results = []

# %% load data
print('Loading data...')
if os.path.exists('./dataset.pkl'):
    frames, truths = pickle.load(open('./dataset.pkl', 'rb'))
else:
    frames, truths = load_all_data()
    pickle.dump([frames, truths], open('./dataset.pkl', 'wb'))

# %% run tuning
print('\nRunning parameter tuning...')

t = trange(max_iter, desc='Highest:', leave=True)
for i in t:
    auc, eer = quick_pass(classifier, frames, truths)
    t.set_description('Current: {:.4f} || Highest: {:.4f}'.format(auc, highest))
    t.refresh()
    classifier.random_update_params()
    if auc > highest:
        highest = auc
        results.append([classifier.weight, auc, eer])
