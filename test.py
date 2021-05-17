import pandas as pd

from classifiers.basic import BasicThresholdClassifer
from data_loader import short_time_feature_loader
from evaluate import get_metrics


FRAME_SIZE = 0.032   # 100ms per frame
FRAME_SHIFT = 0.008  # 40ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)
medfilt_size = 15

# set file paths here
set_path = './wavs/train'
label_path = './data/train_label.txt'

time = pd.read_csv('./time_domain_features.csv')
freq = pd.read_csv('./freq_domain_features.csv')

classifier = BasicThresholdClassifer(time, freq)
frames, labels = short_time_feature_loader(
    data_set_path=set_path,
    label_path=label_path,
    use_window='hamming',
    frame_size=N_FRAME,
    frame_shift=N_SHIFT,
    medfilt_size=medfilt_size,
    bin_mode='coarse'
)

pred = classifier.predict(frames)

auc, eer = get_metrics(pred, labels)
print('Run Finished.')
print('  - AUC: {:.4f}'.format(auc))
print('  - EER: {:.4f}'.format(eer))
