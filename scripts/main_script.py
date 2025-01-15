import scipy.io as sio
from scipy.signal import decimate, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt #for plotting, optional
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json

# Load the path from the config file
with open("config.json", "r") as file:
    config = json.load(file)

mat_file_path = config["data_dir"]

# Load the .mat file
mat_contents = sio.loadmat(f'{mat_file_path}/S1.mat')

# Access the data (assuming it's named 'data' inside the .mat file)
data = mat_contents['data'] #adapt if your data variable name is different

#Data dimensions
print(data.shape) # Check the shape to ensure it matches the README info

# Accessing a specific trial (example):
# electrode index 10, time point 500, target index 5, block index 2
trial_data = data[10, 500, 5, 2] 
print(trial_data)

# Load frequency and phase info
freq_phase = sio.loadmat(f'{mat_file_path}/Freq_Phase.mat')  
print(freq_phase.keys())

# Downsample data
downsample_factor = 4  # 1000 Hz to 250 Hz
data_downsampled = decimate(data, downsample_factor, axis=1, zero_phase=True)

# Apply a bandpass filter to isolate the SSVEP frequency range (e.g., 8–16 Hz).
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

fs = 250  # Sampling rate after downsampling
data_filtered = bandpass_filter(data, 8, 16, fs)

# Normalize each epoch to reduce inter-subject variability:
def normalize_epochs(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

data_normalized = normalize_epochs(data_filtered)


# Canonical Correlation Analysis (CCA) commonly used for SSVEP feature extraction:
""" def cca_reference_signals(freqs, fs, n_harmonics=3, duration=5.5):
    t = np.linspace(0, duration, int(fs * duration))
    ref_signals = []
    for f in freqs:
        signal = []
        for h in range(1, n_harmonics + 1):
            signal.append(np.sin(2 * np.pi * h * f * t))
            signal.append(np.cos(2 * np.pi * h * f * t))
        ref_signals.append(np.array(signal))
    return np.array(ref_signals)
 """

# Canonical Correlation Analysis (CCA) commonly used for SSVEP feature extraction:
def cca_reference_signals(frequencies, fs, n_harmonics=3, duration=5.5):
    """
    Generate sinusoidal reference signals for CCA.
    
    Args:
        frequencies (np.ndarray): Array of target frequencies (e.g., 8-15.8 Hz).
        fs (int): Sampling rate of the EEG data.
        n_harmonics (int): Number of harmonics to include in the reference signals.
        duration (float): Duration of each epoch in seconds.
    
    Returns:
        np.ndarray: Reference signals with shape [num_frequencies, num_signals, num_samples].
    """
    t = np.linspace(0, duration, int(fs * duration))  # Time vector
    reference_signals = []
    
    for f in frequencies.flatten():  # Iterate through each frequency
        harmonic_signals = []
        for h in range(1, n_harmonics + 1):
            harmonic_signals.append(np.sin(2 * np.pi * h * f * t))  # Sinusoidal signal
            harmonic_signals.append(np.cos(2 * np.pi * h * f * t))  # Cosine signal
        reference_signals.append(np.array(harmonic_signals))
    
    return np.array(reference_signals)


""" # Generate reference signals for target frequencies
frequencies = freq_phase['freqs']  # Target frequencies (8-15.8 Hz)
ref_signals = cca_reference_signals(frequencies, fs)

# CCA on a single trial
trial = data_normalized[:, 0, 0]  # Example trial
trial_cca = trial.mean(axis=0)  # Averaging across channels for simplicity

cca = CCA(n_components=1)
cca.fit(ref_signals[0].T, trial_cca.T) """

# Generate reference signals for target frequencies
frequencies = freq_phase['freqs'].flatten()  # Ensure it's a 1D array
ref_signals = cca_reference_signals(frequencies, fs)

# Select a specific trial
block_idx = 0  # First block
target_idx = 0  # First target
trial = data_normalized[:, :, target_idx, block_idx]  # Shape: [64, 1375]

# Extract the reference signals for the target frequency
ref_signals_current = ref_signals[target_idx]  # Shape: [6, 1375] (6 harmonics x 1375 time points)

# Apply CCA
cca = CCA(n_components=1)
cca.fit(trial.T, ref_signals_current.T)  # Inputs: [1375, 64] and [1375, 6]

# Transform and compute correlation
X_c, Y_c = cca.transform(trial.T, ref_signals_current.T)
correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
print(f"Correlation for Target {target_idx}: {correlation}")


# Train-test split, prepare dataset
# Flatten data to shape [num_samples, num_features]
X = data_normalized.reshape(-1, 64 * 1500)
y = np.tile(np.arange(40), 6 * 35)  # Targets for each trial

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier, such as SVM
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize EEG signals or spectral power:
# Plot a single trial
plt.plot(data_filtered[0, :, 0, 0])  # Channel 0, Trial 0
plt.title("Filtered EEG Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.show()
