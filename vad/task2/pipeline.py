import os
import sys
import pickle
import numpy as np
from ..data_loader import spectral_feature_loader
from ..evaluate import get_metrics
from ..classifiers.dualGMM import DualGMMClassifier


def train(
        VADClassifier: DualGMMClassifier,
        train_set_path,
        train_label_path,
        n_frame=512,
        n_shift=128,):
    # load datasets
    print('Loading training set...', file=sys.stderr)
    if os.path.exists('./data/extracted/training_set.pkl'):
        X_train, sample_lengths, Y_train = pickle.load(
            open('./data/extracted/training_set.pkl', 'rb'))
    else:
        X_train, sample_lengths, Y_train = spectral_feature_loader(
            train_set_path, train_label_path,
            frame_size=n_frame, frame_shift=n_shift,)
        pickle.dump(
            [X_train, sample_lengths, Y_train],
            open('./data/extracted/training_set.pkl', 'wb'))

    # convert data
    X_train = X_train.T  # (n_features, n_samples) -> (n_samples, n_features)

    # train the model
    print('Training model...', file=sys.stderr)
    VADClassifier.fit(X_train, Y_train)

    # evaluate on training set
    pred_train_prob = VADClassifier.predict_proba(X_train)[:, 0]
    pred_train = np.where(pred_train_prob >= 0.5, 1, 0)

    auc_prob, eer_prob = get_metrics(pred_train_prob, Y_train)
    auc, eer = get_metrics(pred_train, Y_train)

    print('Done.')
    print(
        '  - AUC on train: {:.4f} | {:.4f}'.format(auc, auc_prob))
    print(
        '  - EER on train: {:.4f} | {:.4f}'.format(eer, eer_prob))


def evaluate(
        VADClassifier: DualGMMClassifier,
        dev_set_path,
        dev_label_path,
        n_frame=512,
        n_shift=128,):
    print('Loading dev set...', file=sys.stderr)
    if os.path.exists('./data/extracted/dev_set.pkl'):
        X_dev, sample_lengths, Y_dev = pickle.load(
            open('./data/extracted/dev_set.pkl', 'rb'))
    else:
        X_dev, sample_lengths, Y_dev = spectral_feature_loader(
            dev_set_path, dev_label_path,
            frame_size=n_frame, frame_shift=n_shift)
        pickle.dump(
            [X_dev, sample_lengths, Y_dev],
            open('./data/extracted/dev_set.pkl', 'wb'))

    X_dev = X_dev.T

    print('Evaluating model...', file=sys.stderr)
    pred_dev_prob = VADClassifier.predict_proba(X_dev)[:, 0]
    pred_dev = np.where(pred_dev_prob >= 0.5, 1, 0)

    auc_prob, eer_prob = get_metrics(pred_dev_prob, Y_dev)
    auc, eer = get_metrics(pred_dev, Y_dev)

    print('Done')
    print(
        '  - AUC on dev:   {:.4f} | {:.4f}'.format(auc, auc_prob))
    print(
        '  - EER on dev:   {:.4f} | {:.4f}'.format(eer, eer_prob))
