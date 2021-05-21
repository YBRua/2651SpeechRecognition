import pickle
import vad.task2.pipeline as task2Ppl
from vad.classifiers.dualGMM import DualGMMClassifier

train_set_path = './wavs/train'
train_label_path = './data/train_label.txt'
dev_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'

VADClassifier = DualGMMClassifier(
    n_components=3,
    covariance_type='full',
    max_iter=500,
    verbose=1,
    random_state=1919810,
)

pickle.dump(VADClassifier, open('model_1.pkl', 'wb'))

task2Ppl.train(VADClassifier, train_set_path, train_label_path)
task2Ppl.evaluate(VADClassifier, dev_set_path, dev_label_path)
