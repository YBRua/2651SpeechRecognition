from collections import namedtuple
import numpy as np


Voiced = namedtuple(
    'Voiced',
    [
        'mean_magnitude', 'min_magnitude',
        'mean_energy', 'min_energy',
        'mean_zcr', 'min_zcr',
        'mean_lowfreq', 'min_lowfreq',
        'mean_medfreq', 'min_medfreq',
        'mean_highfreq', 'min_highfreq',
    ])

Unvoiced = namedtuple(
    'Unvoiced',
    [
        'mean_magnitude', 'max_magnitude',
        'mean_energy', 'max_energy',
        'mean_zcr', 'max_zcr',
        'mean_lowfreq', 'max_lowfreq',
        'mean_medfreq', 'max_medfreq',
        'mean_highfreq', 'max_highfreq',
    ]
)


ScoreWeight = namedtuple(
    'ScoreWeight',
    [
        'mag', 'enr', 'zcr',
        'low', 'med', 'high',
        'state_weight',
        'primary_passes', 'secondary_passes'
    ]
)


class BasicThresholdClassifer():
    def __init__(self, time, freq, score_weight=None):
        self.voiced = Voiced(
            time['Voiced Magnitude'].mean(), time['Voiced Magnitude'].min(),
            time['Voiced Energy'].mean(), time['Voiced Energy'].min(),
            time['Voiced ZCR'].mean(), time['Voiced ZCR'].min(),
            freq['Voiced LowFreq'].mean(), freq['Voiced LowFreq'].mean(),
            freq['Voiced MedFreq'].mean(), freq['Voiced MedFreq'].min(),
            freq['Voiced HighFreq'].mean(), freq['Voiced HighFreq'].min()
        )
        self.unvoiced = Unvoiced(
            time['Unvoiced Magnitude'].mean(), time['Unvoiced Magnitude'].max(),
            time['Unvoiced Energy'].mean(), time['Unvoiced Energy'].max(),
            time['Unvoiced ZCR'].mean(), time['Unvoiced ZCR'].max(),
            freq['Unvoiced LowFreq'].mean(), freq['Unvoiced LowFreq'].max(),
            freq['Unvoiced MedFreq'].mean(), freq['Unvoiced MedFreq'].max(),
            freq['Unvoiced HighFreq'].mean(), freq['Unvoiced HighFreq'].max()
        )

        self.mag_boundary =\
            (self.voiced.min_magnitude + self.unvoiced.max_magnitude) / 2
        self.energy_boundary =\
            (self.voiced.min_energy + self.unvoiced.max_energy) / 2
        self.zcr_boundary =\
            (self.voiced.min_zcr + self.unvoiced.max_zcr) / 2
        self.lowfreq_boundary =\
            (self.voiced.min_lowfreq + self.unvoiced.max_lowfreq) / 2
        self.medfreq_boundary =\
            (self.voiced.min_medfreq + self.unvoiced.max_medfreq) / 2
        self.highfreq_boundary =\
            (self.voiced.min_highfreq + self.unvoiced.max_highfreq) / 2

        self.state = 0

        if score_weight is not None:
            self.weight = score_weight
        else:
            self.weight = ScoreWeight(
                mag=2,
                enr=2,
                zcr=0,
                low=1,
                med=4,
                high=1,
                state_weight=2,
                primary_passes=5,
                secondary_passes=6
            )

    def _check_primary_features(self, x):
        passes = 0
        if x[0] > self.mag_boundary:
            passes += self.weight.mag
        if x[1] > self.energy_boundary:
            passes += self.weight.enr
        if x[4] > self.medfreq_boundary:
            passes += self.weight.med

        return passes + self.state * self.weight.state_weight

    def _check_secondary_features(self, x, passes):
        if x[2] > self.zcr_boundary:
            passes += self.weight.zcr
        if x[3] > self.lowfreq_boundary:
            passes += self.weight.low
        if x[5] > self.highfreq_boundary:
            passes += self.weight.high
        return passes

    def random_update_params(self):
        param_list = list(self.weight)
        for i in range(len(param_list)):
            pertub = np.random.randn() * 0.1
            if pertub + param_list[i] >= 0:
                param_list[i] += pertub
        self.weight = ScoreWeight(*param_list)

    def pred_one_frame(self, x):
        primal_passes = self._check_primary_features(x)
        secondary_passes = self._check_secondary_features(x, primal_passes)
        if primal_passes >= self.weight.primary_passes:
            self.state = 1
            return 1
        elif secondary_passes >= self.weight.secondary_passes:
            self.state = 1
            return 1
        else:
            self.state = 0
            return 0

    def pred(self, x):
        pred = []
        for frame in x:
            pred.append(self.pred_one_frame(frame))

        return pred
