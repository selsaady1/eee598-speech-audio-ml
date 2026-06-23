# Speech & Audio ML — Keyword Spotting

> Lightweight keyword spotting from scratch using classical DSP and machine learning

![MATLAB](https://img.shields.io/badge/MATLAB-0a7?style=flat-square) ![Python](https://img.shields.io/badge/Python-0a7?style=flat-square) ![librosa](https://img.shields.io/badge/librosa-0a7?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-0a7?style=flat-square) ![NumPy](https://img.shields.io/badge/NumPy-0a7?style=flat-square) ![pandas](https://img.shields.io/badge/pandas-0a7?style=flat-square) ![SciPy](https://img.shields.io/badge/SciPy-0a7?style=flat-square) ![matplotlib](https://img.shields.io/badge/matplotlib-0a7?style=flat-square) ![seaborn](https://img.shields.io/badge/seaborn-0a7?style=flat-square) ![Google Speech Commands v2 dataset](https://img.shields.io/badge/Google_Speech_Commands_v2_dataset-0a7?style=flat-square) 

### 🌐 Live project page → **https://selsaady1.github.io/eee598-speech-audio-ml/**

## Overview
A keyword spotting (KWS) system that recognizes the spoken words 'yes', 'no', 'stop', and 'go' plus a silence/unknown class, built entirely with classical signal processing instead of deep learning. The goal was to show that hand-crafted time- and frequency-domain features paired with traditional classifiers can hit competitive accuracy while staying small and fast enough for embedded or System-on-Chip deployment. Completed for ASU's EEE 598 Speech and Audio Processing course with a project partner.

**Highlight:** 84.67% test accuracy

## Key Achievements
- Achieved 84.67% test accuracy on the 5-class task using an SVM (RBF kernel, ECOC) with a 30-dimensional feature vector, exceeding the project's 75% target
- Built a full audio-to-decision pipeline from scratch (resampling to 16 kHz, energy-based silence trimming at -45 dB, peak normalization, 1.0 s length standardization) in both MATLAB and Python
- Engineered a 30-D feature vector combining short-time energy, zero-crossing rate, and 13 MFCCs (mean and std per coefficient via STFT -> Mel filterbank -> log -> DCT), with delta/delta-delta features in the Python pipeline
- Ran a comparative study and feature ablation (k-NN vs SVM; MFCC-only vs MFCC+Energy+ZCR), finding SVM beat k-NN by ~5.2% and MFCCs alone already captured most discriminative power (+0.6% from time-domain features)
- Evaluated noise robustness across SNR levels with both white and real background noise, showing graceful degradation (84.67% clean down through ~57-69% at 20 dB) and >50% accuracy to roughly 10 dB SNR
- Demonstrated embedded feasibility: model under 1 MB, ~120 bytes per clip, and inference under 10 ms per 1-second clip, suitable for ARM Cortex-M4-class processors

## Approach
Audio from the Google Speech Commands v2 dataset was preprocessed (16 kHz resample, silence trimming, normalization, fixed 1 s length) and split into train/validation/test sets. Features were extracted over 20-25 ms Hamming windows with a 10 ms hop: short-time energy, zero-crossing rate, and MFCCs, aggregated to a fixed-length per-clip vector. k-NN (k in {1,3,5,7}) and SVM (RBF kernel, grid-searched kernel scale and box constraint) classifiers were trained and tuned on the validation set, then evaluated on a held-out test set and under added noise. The system was implemented in MATLAB and re-implemented as a four-phase Python pipeline (librosa, scikit-learn).

## Tools & Technologies
- MATLAB
- Python
- librosa
- scikit-learn
- NumPy
- pandas
- SciPy
- matplotlib
- seaborn
- Google Speech Commands v2 dataset
- MFCC / STFT / DCT (DSP)
- SVM
- k-NN

## Repository Structure
```
.gitignore
LICENSE
README.md
docs/EEE 598 - Final Project Report-2.docx.pdf
docs/EEE598_Final_Project_Proposal.pdf
docs/FinalProject.docx
docs/Kws Project Plan (1).pdf
docs/Lightweight Keyword Spotting Using Energy and MFCC Features for Embedded Deployment (1).pdf
images/preview.png
src/Autocorr_Demo.m
src/EEE598_Final_Project.m
src/Filtering_Demo.m
src/Fourier_Demo.m
src/OLA_Demo.m
src/Phase1_Preprocessing_Script.py
src/Phase2_Feature_Extraction.py
src/Phase3_Classifier_Training.py
src/Phase4_Testing_Final_Evaluation.py
src/Quantization_Demo.m
src/Spectrogram_Demo.m
src/Speech_AutocorrAMDF_Demo.m
src/Speech_Quantization_Demo.m
src/ZCR_Energy_Demo.m
src/cepstrum.m
src/cross_synthesis.m
src/plot_fft.m
src/source_filter.m
```

## Results
The best configuration (SVM with MFCC+Energy+ZCR features) reached 84.67% test accuracy on five classes, with the 'silence' class near-perfect (99.5%) and the main confusion between acoustically similar 'no' and 'go'. Under noise the system degraded gracefully, staying above 50% accuracy down to roughly 10 dB SNR. Full metrics, confusion matrices, and noise tables are in the final report.

## Deliverable
See [`docs/EEE 598 - Final Project Report-2.docx.pdf`](docs/EEE%20598%20-%20Final%20Project%20Report-2.docx.pdf).

## License
MIT — see [`LICENSE`](LICENSE).

---
_Part of [Saif Elsaady's engineering portfolio](https://selsaady1.github.io/portfolio/). Deliverables only — routine homework/quizzes/exams excluded._