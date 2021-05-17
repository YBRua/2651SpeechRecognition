import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io.wavfile as wavfile

from features.short_time_features import short_time_feature_extractor
from vad_utils import prediction_to_vad_label
from classifiers.basic import BasicThresholdClassifer, ScoreWeight


FRAME_SIZE = 0.032   # 100ms per frame
FRAME_SHIFT = 0.008  # 40ms frame-shift
SAMPLE_RATE = 16000  # 16kHz sample rate
N_FRAME = int(FRAME_SIZE * SAMPLE_RATE)
N_SHIFT = int(FRAME_SHIFT * SAMPLE_RATE)

test_set_path = './wavs/test'

# please make sure that the following csv files exist
time = pd.read_csv('./time_domain_features.csv')
freq = pd.read_csv('./freq_domain_features.csv')

medfilt_size = 15

# optimal weight found by parameter tuning
optimal_weight = ScoreWeight(
    2.3660515227, 2.20055434, 0.4418205904,
    0.3016455199, 4.3315168325, 1.7074504699,
    3.4955110421, 5.6886246655, 6.6389220191
)

# initialize classifier using data from dev set
classifier = BasicThresholdClassifer(time, freq, optimal_weight)

with open('./test_label_task1.txt', 'w') as output:
    # load data from test set
    for root, dirs, files in os.walk(test_set_path):
        for f in tqdm(files):
            if '.wav' in f:
                output.write(f.replace('.wav', ' '))
                rate, data = wavfile.read(os.path.join(test_set_path, f))
                data = np.array(data, dtype=float)
                data -= np.mean(data)   # remove dc-offset
                data /= 32767           # normalize

                # feature extraction
                frames = short_time_feature_extractor(
                    data, medfilt_size=medfilt_size).T
                # predict labels
                pred = classifier.predict(frames)
                result = prediction_to_vad_label(pred)
                output.write(result)
                output.write('\n')
