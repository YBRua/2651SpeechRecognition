# AI2651 Intelligent Speech Recognition
Projects and assignments for AI2651 Intelligent Speech Recognition.
## Project 1: Voice Activity Detection
### Task 1: Simple Classifier
Detect voice activity using short-time features of voice signals and basic linear classifiers.

#### Progress
> Finished!

- [x] short-time energy
- [x] short-time magnitude
- [x] short-time zero-crossing rate
- [x] basic short-time feature extraction
- [x] short-time Fourier transform
- [x] feature extraction pipeline
- [x] estimate average values of features, to be used as thresholds
- [x] find and implement a proper classifier
- [x] Reference book: Bayes-based VAD
- [x] Parameter tuning
- [x] How to use ZCR and Low-freq Energy
- [x] doc strings for functions
- [x] Technical report.

### Task 2: Spectral Features & Statistic Classifiers
Detect voice activity using spectral features of voice signals and statistic machine learning classifiers.

#### Progress
- [x] MFCC feature extraction
- [x] GMMHMM test run.
  - GMM: AUC 0.86 (trained on dev set)
  - AUC 0.9173 when using `predict_proba`
- [x] GMM test run.
  - GMMHMM: AUC 0.87 (trained on dev set)
  - AUC 0.96 when using `predict_proba`
- [x] DualGMM classifier.
  - 0.94 AUC (trained on dev set)
- [x] Optimize file structure.
- [x] Training and evaluation.
  - [ ] Feature Engineering
- [ ] Model selection.
- [ ] Run on test set.
- [ ] Technical Report.

#### Memo
- Feature extraction on training set is SLOW.
- `n_mfcc` matters.
- `delta` matters. 1st- and 2nd-order or 2nd- and 3rd-order? Or all three orders?