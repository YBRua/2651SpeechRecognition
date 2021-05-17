import os
import sys
import pickle
import numpy as np
from data_loader import spectral_feature_loader
from evaluate import get_metrics
from classifiers.dualGMM import DualGMMClassifier

train_set_path = './wavs/train'
dev_set_path = './wavs/dev'
train_label_path = './data/train_label.txt'
dev_label_path = './data/dev_label.txt'

# load datasets
print('Loading training set...', file=sys.stderr)
if os.path.exists('./training_set.pkl'):
    X_train, sample_lengths, Y_train = pickle.load(
        open('./training_set.pkl', 'rb'))
else:
    X_train, sample_lengths, Y_train = spectral_feature_loader(
        train_set_path, train_label_path)
    pickle.dump([X_train, sample_lengths, Y_train],
                open('./training_set.pkl', 'wb'))

print('Loading dev set...', file=sys.stderr)
if os.path.exists('./dev_set.pkl'):
    X_dev, sample_lengths, Y_dev = pickle.load(open('./dev_set.pkl', 'rb'))
else:
    X_dev, sample_lengths, Y_dev = spectral_feature_loader(
        dev_set_path, dev_label_path)
    pickle.dump([X_dev, sample_lengths, Y_dev], open('./dev_set.pkl', 'wb'))

VADClassifier = DualGMMClassifier(
    n_components=3,
    covariance_type='full',
    max_iter=500,
    verbose=0,
    random_state=1919810,
)

# convert data
X_train = X_train.T  # (n_features, n_samples) -> (n_samples, n_features)
X_dev = X_dev.T

X_train_voiced = X_train[Y_train == 1]
X_train_unvoiced = X_train[Y_train == 0]

# train the model
print('Training model...', file=sys.stderr)
VADClassifier.fit(X_train_voiced, X_train_unvoiced, Y_train)
pickle.dump(VADClassifier, open('model_1.pkl', 'wb'))

# %% evaluate
print('Evaluating...')
pred_train_prob = VADClassifier.predict_proba(X_train)[:, 0]
pred_dev_prob = VADClassifier.predict_proba(X_dev)[:, 0]
pred_train = np.where(pred_train_prob >= 0.5, 1, 0)
pred_dev = np.where(pred_dev_prob >= 0.5, 1, 0)

auc_tr_prob, eer_tr_prob = get_metrics(pred_train_prob, Y_train)
auc_dv_prob, eer_dv_prob = get_metrics(pred_dev_prob, Y_dev)
auc_tr, eer_tr = get_metrics(pred_train, Y_train)
auc_dv, eer_dv = get_metrics(pred_dev, Y_dev)

print('Finished.')
print(
    '  - AUC on train: {:.4f} | {:.4f}'.format(auc_tr*100, auc_tr_prob*100))
print(
    '  - EER on train: {:.4f} | {:.4f}'.format(eer_tr*100, eer_tr_prob*100))
print(
    '  - AUC on dev:   {:.4f} | {:.4f}'.format(auc_dv*100, auc_dv_prob*100))
print(
    '  - EER on dev:   {:.4f} | {:.4f}'.format(eer_dv*100, eer_dv_prob*100))
