# Voice Activity Detection DevLog.

## Phase-One: Short-time-feature-based VAD
In the first part of this project, we implements a basic voice activity detector using short-time features of speech signals and basic threshold classifiers.

### Progress
- [x] short-time energy
- [x] short-time magnitude
- [x] short-time zero-crossing rate
- [x] basic short-time feature extraction
- [x] short-time Fourier transform
- [x] feature extraction pipeline
### ToDo
- [ ] estimate average values of features, to be used as thresholds
- [ ] find and implement a proper classifier
- [ ] Reference book: Bayes-based VAD
- [ ] denoising?
- [ ] doc strings for functions?

### Problems and Notes
- some speech files have DC offset. Eg. `472-130755-0013.wav`
- cannot take log of energy due to divide-by-zero exceptions
- The usage of ZCR and frequency-domain features is unclear