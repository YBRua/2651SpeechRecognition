import os
import pickle
from vad.data_loader import spectral_feature_loader

train_set_path = './wavs/train'
dev_set_path = './wavs/dev'
train_label_path = './data/train_label.txt'
dev_label_path = './data/dev_label.txt'

n_mfcc_list = [8, 12, 15, 20]

# all deltas
for n_mfcc in n_mfcc_list:
    if not os.path.exists(''.join(['train_123_', str(n_mfcc), '.pkl'])):
        X_train, sample_lengths, Y_train = spectral_feature_loader(
            train_set_path, train_label_path,
            n_mfcc=n_mfcc,
            use_first_order=True, use_third_order=True)

        pickle.dump(
            [X_train, sample_lengths, Y_train],
            open(''.join(['train_123_', str(n_mfcc), '.pkl']), 'wb'))

    if not os.path.exists(''.join(['dev_123_', str(n_mfcc), '.pkl'])):
        X_dev, sample_lengths_dev, Y_dev = spectral_feature_loader(
            dev_set_path, dev_label_path,
            n_mfcc=n_mfcc,
            use_first_order=True, use_third_order=True)

        pickle.dump(
            [X_dev, sample_lengths_dev, Y_dev],
            open(''.join(['dev_123_', str(n_mfcc), '.pkl']), 'wb'))
