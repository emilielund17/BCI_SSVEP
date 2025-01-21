# BCI_SSVEP
This is a repository for a project in 22053
The scripts are four different deep learning models classifing SSVEP BCI data; a CNN and ResNet modelleing on 64 (CNN64, ResNet64) and 9 
(CNN9, ResNet9) channel eeg data respectivley. 
The evaluation metrics, accuracy, and ITR, is saved and stores in the new_model_metric.txt 


# **SSVEP-Based BCI Classification**

This repository contains scripts for classifying SSVEP signals using CNN and ResNet architectures on a benchmark EEG dataset. The project evaluates the performance of models with all 64 electrodes and a reduced set of 9 task-relevant electrodes.

## **Features**
- EEG Preprocessing: Bandpass filtering (7–90 Hz) and normalization.
- Feature Extraction: FFT-based harmonic features for SSVEP classification.
- Model Architectures: CNN and ResNet with 64 and 9 electrodes.
- Performance Metrics: Classification accuracy and Information Transfer Rate (ITR).

## **Scripts**
1. **CNN_64_Electrodes.py**: CNN with all 64 EEG electrodes.
2. **CNN_9_Electrodes.py**: CNN with 9 task-relevant electrodes.
3. **ResNet_64_Electrodes.py**: ResNet with all 64 electrodes.
4. **ResNet_9_Electrodes.py**: ResNet with 9 electrodes, achieving the highest accuracy (93.25%) and ITR.

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ssvep-bci-classification.git

