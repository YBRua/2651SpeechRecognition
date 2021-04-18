from collections import namedtuple


Voiced = namedtuple(
    'Voiced',
    [
        'mean_magnitude', 'min_magnitude',
        'mean_energy', 'min_energy',
        'mean_zcr',
        'mean_lowfreq',
        'mean_medfreq', 'min_medfreq',
        'mean_highfreq', 'min_highfreq',
    ])

Unvoiced = namedtuple(
    'Unvoiced',
    [
        'mean_magnitude', 'max_magnitude',
        'mean_energy', 'max_energy',
        'mean_zcr',
        'mean_lowfreq',
        'mean_medfreq', 'max_medfreq',
        'mean_highfreq', 'max_highfreq',
    ]
)


class BasicThresholdClassifer():
    def __init__(self, time, freq):
        self.voiced = Voiced(
            time['Voiced Magnitude'].mean(), time['Voiced Magnitude'].min(),
            time['Voiced Energy'].mean(), time['Voiced Energy'].min(),
            time['Voiced ZCR'].mean(),
            freq['Voiced LowFreq'].mean(),
            freq['Voiced MedFreq'].mean(), freq['Voiced MedFreq'].min(),
            freq['Voiced HighFreq'].mean(), freq['Voiced HighFreq'].min()
        )
        self.unvoiced = Unvoiced(
            time['Unvoiced Magnitude'].mean(), time['Unvoiced Magnitude'].max(),
            time['Unvoiced Energy'].mean(), time['Unvoiced Energy'].max(),
            time['Unvoiced ZCR'].mean(),
            freq['Unvoiced LowFreq'].mean(),
            freq['Unvoiced MedFreq'].mean(), freq['Unvoiced MedFreq'].max(),
            freq['Unvoiced HighFreq'].mean(), freq['Unvoiced HighFreq'].max()
        )

        self.mag_boundary =\
            (self.voiced.min_magnitude + self.unvoiced.max_magnitude) / 2
        self.energy_boundary =\
            (self.voiced.min_energy + self.unvoiced.max_energy) / 2
        self.medfreq_boundary =\
            (self.voiced.min_medfreq + self.unvoiced.max_medfreq) / 2
        self.highfreq_boundary =\
            (self.voiced.min_highfreq + self.unvoiced.max_highfreq) / 2

        self.state = 0

    def _check_primal_features(self, x):
        passes = 0
        if x[0] > self.mag_boundary:
            passes += 1
        if x[1] > self.energy_boundary:
            passes += 1
        if x[4] > self.medfreq_boundary:
            passes += 2

        return passes

    def _check_secondary_features(self, x, passes):
        if x[4] > self.highfreq_boundary:
            passes += 1
        return passes + self.state

    def pred(self, x):
        primal_passes = self._check_primal_features(x)
        secondary_passes = self._check_secondary_features(x, primal_passes)
        if primal_passes >= 3:
            self.state = 1
            return 1
        elif secondary_passes >= 3:
            self.state = 1
            return 1
        else:
            self.state = 0
            return 0
