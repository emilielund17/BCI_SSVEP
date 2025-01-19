import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import json
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Preprocessing function
def preprocess_eeg(eeg_data, sampling_rate):
    # Bandpass filter (e.g., 7-90 Hz)
    nyquist = sampling_rate / 2
    low, high = 7 / nyquist, 90 / nyquist
    b, a = signal.cheby1(N=4, rp=0.5, Wn=[low, high], btype='band')
    filtered_eeg = signal.filtfilt(b, a, eeg_data, axis=1)

    # Normalize per channel
    normalized_eeg = (filtered_eeg - np.mean(filtered_eeg, axis=1, keepdims=True)) / np.std(filtered_eeg, axis=1, keepdims=True)

    return normalized_eeg

# Feature extraction function
def extract_features(eeg_data, sampling_rate, num_harmonics=3):
    fft_data = np.abs(np.fft.rfft(eeg_data, axis=1))
    freqs = np.fft.rfftfreq(eeg_data.shape[1], d=1 / sampling_rate)

    # Select features at fundamental and harmonic frequencies
    features = []
    for freq in np.arange(8, 16, 0.2):  # Stimulus frequencies
        for h in range(1, num_harmonics + 1):
            idx = np.argmin(np.abs(freqs - (freq * h)))
            features.append(fft_data[:, idx])

    return np.array(features).T

# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

data_dir = config["data_dir"]  # Folder containing .mat files
sampling_rate = 250  # Set sampling rate

# Preprocess and extract features for all subjects
X, y = [], []
frequencies = np.arange(8, 16, 0.2)

# Loop through all subject files in the folder
for mat_file in os.listdir(data_dir):
    if mat_file.endswith(".mat") and mat_file.startswith('S'): 
        print(f"Processing file: {mat_file}")
        
        # Load the .mat file
        mat_contents = sio.loadmat(os.path.join(data_dir, mat_file))
        eeg_data = mat_contents['data']  # Shape: [64, 1500, 40, 6]
        
        # Loop through blocks and trials
        for block_idx in range(eeg_data.shape[3]):
            for trial_idx in range(40):
                raw_trial = eeg_data[:, :, trial_idx, block_idx]
                preprocessed_trial = preprocess_eeg(raw_trial, sampling_rate)
                features = extract_features(preprocessed_trial, sampling_rate)
                X.append(features)
                y.append(trial_idx // 6)  # Assuming 6 blocks per frequency

X = np.array(X)
# Convert one-hot encoded labels back to class indices
y = np.argmax(y, axis=1)

# Split data into train, test, and validation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# Flatten the data for SVM (as SVM expects 2D input)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_eval = X_eval.reshape(X_eval.shape[0], -1)

# Train SVM Classifier
print("Training SVM...")
start_time = time.time()
svm_classifier = SVC(kernel='linear', C=1.0, probability=True)  # Linear kernel
svm_classifier.fit(X_train, y_train)
training_time = time.time() - start_time

# Evaluate the SVM on validation data
y_pred_eval = svm_classifier.predict(X_eval)
eval_accuracy = accuracy_score(y_eval, y_pred_eval)
print(f"Evaluation Accuracy: {eval_accuracy * 100:.2f}%")

# Calculate Information Transfer Rate (ITR)
def calculate_itr(T, N, P):
    """
    Calculate Information Transfer Rate (ITR) in bits per minute.

    Parameters:
    - T: Trial duration (seconds)
    - N: Number of classes (targets)
    - P: Accuracy (fraction, e.g., 0.85 for 85%)

    Returns:
    - ITR in bits per minute
    """
    if P == 0 or P == 1:
        return 0  # ITR is 0 if accuracy is perfect (P=1) or zero (P=0)
    
    itr = (60 / T) * (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))
    return itr

T = 8  # Trial duration in seconds
N = len(frequencies)  # Number of classes
P = eval_accuracy  # Model's accuracy on evaluation set
itr = calculate_itr(T, N, P)
print(f"Information Transfer Rate (ITR): {itr:.2f} bits/minute")

# Save metrics to a text file
output_file = "model_metrics_svm.txt"
with open(output_file, "a") as f:
    f.write("\n--- Results from SVM Classifier ---\n")
    f.write(f"Evaluation Accuracy: {eval_accuracy * 100:.2f}%\n")
    f.write(f"Information Transfer Rate (ITR): {itr:.2f} bits/minute\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write("-" * 40 + "\n")

print(f"Metrics appended to {output_file}")
