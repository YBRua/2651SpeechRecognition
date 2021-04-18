# Voice Activity Detection DevLog.

## Phase-One: Short-time-feature-based VAD
In the first part of this project, we implements a basic voice activity detector using short-time features of speech signals and basic threshold classifiers.

### Progress
**A very basic classifier prototype has been completed**
- [x] short-time energy
- [x] short-time magnitude
- [x] short-time zero-crossing rate
- [x] basic short-time feature extraction
- [x] short-time Fourier transform
- [x] feature extraction pipeline
- [x] estimate average values of features, to be used as thresholds
- [x] find and implement a proper classifier
- [x] Reference book: Bayes-based VAD
### ToDo
- [ ] Write technical report
- [ ] Parameter tuning
- [ ] How to use ZCR and Low-freq Energy?
- [ ] denoising?
- [ ] doc strings for functions?

### Problems and Notes
- some speech files have DC offset. Eg. `472-130755-0013.wav`
- cannot take log of energy due to divide-by-zero exceptions
- The usage of ZCR and frequency-domain features is unclear
- 717471 out of 878694 labels are positive. Positive label frequency: 0.82.
- Bayesian classifier somehow failed. Why?