# %% import libraries and initialize
from classifiers.basic import BasicThresholdClassifer, ScoreWeight
from classification import load_all_data
from short_time_analysis import feature_analysis
from vad_utils import read_label_from_file
from evaluate import get_metrics


FRAME_SIZE = 0.032   # 100ms per frame
FRAME_SHIFT = 0.008  # 40ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

# set file paths here
dev_set_path = './wavs/dev'
dev_label_path = './data/dev_label.txt'

medfilt_size = 15
optimal_weight = ScoreWeight(
    2.3660515227, 2.20055434, 0.4418205904,
    0.3016455199, 4.3315168325, 1.7074504699,
    3.4955110421, 5.6886246655, 6.6389220191
)
# %% feature analysis
labels = read_label_from_file(dev_label_path)
time, freq = feature_analysis(
    dev_set_path, labels,
    N_FRAME, N_SHIFT,
    medfilt_size=medfilt_size
)

time.to_csv('./time_domain_features.csv', index=False)
freq.to_csv('./freq_domain_features.csv', index=False)

# %% construct model and do classification
classifier = BasicThresholdClassifer(time, freq, optimal_weight)
frames, labels = load_all_data(
    data_set_path=dev_set_path,
    label_path=dev_label_path,
    use_window='hamming',
    frame_size=N_FRAME,
    frame_shift=N_SHIFT,
    medfilt_size=medfilt_size,
    bin_mode='coarse'
)
pred = classifier.pred(frames)

auc, eer = get_metrics(pred, labels)
print('Run Finished.')
print('  - AUC: {:.4f}'.format(auc))
print('  - EER: {:.4f}'.format(eer))
