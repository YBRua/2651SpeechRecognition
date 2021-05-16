# AI2651 Intelligent Speech Recognition
Projects and assignments for AI2651 Intelligent Speech Recognition.
## Project 1: Voice Activity Detection
### Task 1: Simple Classifier
Detect voice activity using short-time features of voice signals and basic linear classifiers.

### Task 2: Spectral Features & Statistic Classifiers
Detect voice activity using spectral features of voice signals and statistic machine learning classifiers.

#### Schedule
- [x] MFCC feature extraction
- [x] GMMHMM test run.
  - GMM: AUC 0.86 (trained on dev set)
  - AUC 0.9173 when using `predict_proba`
- [x] GMM test run.
  - GMMHMM: AUC 0.87 (trained on dev set)
  - AUC 0.96 when using `predict_proba`
- [ ] DualGMM classifier.
- [ ] Model selection.
- [ ] Run on test set.
- [ ] Technical Report.