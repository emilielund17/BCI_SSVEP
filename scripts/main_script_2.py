import scipy.io as sio
from scipy.signal import decimate, cheby1, filtfilt
import numpy as np
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

# Filter bank design
def filter_bank(data, fs, sub_bands):
    """
    Apply filter bank to EEG data.
    Args:
        data (np.ndarray): EEG data [channels, time points].
        fs (int): Sampling frequency.
        sub_bands (list): List of frequency band tuples [(low, high), ...].
    Returns:
        np.ndarray: Filtered data [num_sub_bands, channels, time_points].
    """
    filtered_data = []
    for band in sub_bands:
        low, high = band
        nyquist = 0.5 * fs
        low = low / nyquist
        high = high / nyquist
        b, a = cheby1(N=4, rp=0.5, Wn=[low, high], btype='band')
        filtered_data.append(filtfilt(b, a, data, axis=1))
    return np.array(filtered_data)  # Shape: [num_sub_bands, channels, time_points]

# Generate sinusoidal reference signals for CCA
def generate_reference_signals(frequencies, fs, n_harmonics=5, duration=5.5):
    """
    Generate sinusoidal reference signals.
    Args:
        frequencies (np.ndarray): Target frequencies.
        fs (int): Sampling rate.
        n_harmonics (int): Number of harmonics.
        duration (float): Duration of the trial in seconds.
    Returns:
        np.ndarray: Reference signals [num_targets, 2 * n_harmonics, num_samples].
    """
    t = np.linspace(0, duration, int(fs * duration))
    reference_signals = []
    for f in frequencies.flatten():
        harmonic_signals = []
        for h in range(1, n_harmonics + 1):
            harmonic_signals.append(np.sin(2 * np.pi * h * f * t))
            harmonic_signals.append(np.cos(2 * np.pi * h * f * t))
        reference_signals.append(np.array(harmonic_signals))
    return np.array(reference_signals)

""" # Segment data into epochs
def segment_data(data, fs, pre_stimulus=0.5, post_stimulus=5.5): """
    """
    Segment continuous EEG data into epochs.
    Args:
        data (np.ndarray): Continuous EEG data [channels, time points].
        fs (int): Sampling rate.
        pre_stimulus (float): Pre-stimulus duration in seconds.
        post_stimulus (float): Post-stimulus duration in seconds.
    Returns:
        np.ndarray: Segmented data [trials, channels, time_points].
    """
    """ epoch_length = int((pre_stimulus + post_stimulus) * fs) 
    num_trials = data.shape[1] // epoch_length
    segmented_data = np.array([
        data[:, i * epoch_length:(i + 1) * epoch_length]
        for i in range(num_trials)
    ])
    return segmented_data """

# Normalize each trial
def normalize_epochs(data):
    """
    Normalize EEG data by subtracting the mean and dividing by the standard deviation.
    Args:
        data (np.ndarray): EEG data [trials, channels, time points].
    Returns:
        np.ndarray: Normalized EEG data.
    """
    normalized_data = []
    for trial in data:
        mean = np.mean(trial, axis=1, keepdims=True)
        std = np.std(trial, axis=1, keepdims=True)
        normalized_data.append((trial - mean) / std)
    return np.array(normalized_data)

    

# Process a single trial using FBCCA
def process_trial_fbcca(trial, reference_signals, sub_bands):
    """
    Perform FBCCA on a single trial.
    Args:
        trial (np.ndarray): EEG trial data [num_sub_bands, channels, time_points].
        reference_signals (np.ndarray): Reference signals for CCA.
        sub_bands (list): Sub-bands used for filtering.
    Returns:
        int: Predicted target index.
    """
    max_corr = 0
    best_target = -1

    for target_idx, ref in enumerate(reference_signals):
        weighted_corr = 0

        for band_idx, _ in enumerate(sub_bands):
            sub_band_trial = trial[band_idx]
            ref = ref[:, :sub_band_trial.shape[1]]  # Match length

            cca = CCA(n_components=1)
            cca.fit(sub_band_trial.T, ref.T)
            X_c, Y_c = cca.transform(sub_band_trial.T, ref.T)
            corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]

            weight = 1 / (band_idx + 1)  # Weight for sub-band
            weighted_corr += weight * (corr ** 2)

        if weighted_corr > max_corr:
            max_corr = weighted_corr
            best_target = target_idx

    return best_target

# Main pipeline
def main():
    data_dir = config["data_dir"]  # Replace with your data folder path
    num_subjects = 35
    num_targets = 40
    fs = 250  # Sampling frequency

    # Define sub-bands for filter bank
    sub_bands = [(low, low + 8) for low in range(8, 88, 8)]

    # Load frequency and phase information
    freq_phase = sio.loadmat(f'{data_dir}/Freq_Phase.mat')
    frequencies = freq_phase['freqs']  # Target frequencies

    # Generate reference signals
    reference_signals = generate_reference_signals(frequencies, fs)

    # Prepare dataset
    X = []
    y = []

    for subject_id in range(1, num_subjects + 1):
        print(f"Processing subject {subject_id}...")
        mat_contents = sio.loadmat(f'{data_dir}/S{subject_id}.mat')
        data = mat_contents['data']  # Continuous EEG data [channels, time points]

        # Downsample
        data = decimate(data, 4, axis=1, zero_phase=True)  # Downsample to 250 Hz
        print(f"Data shape after downsampling: {data.shape}")

        # Segment data into epochs
        data_epochs = segment_data(data, fs)
        print(f"Segmented data shape: {data_epochs.shape}")

        # Normalize epochs
        data_normalized = normalize_epochs(data_epochs)
        print(f"Normalized data shape: {data_normalized.shape}")

        # Apply filter bank to each epoch
        data_fb = np.array([
            filter_bank(epoch, fs, sub_bands) for epoch in data_normalized
        ])  # Shape: [trials, num_sub_bands, channels, time_points]
        print(f"Filter bank data shape: {data_fb.shape}")

        # Process each trial
        for trial_idx, trial in enumerate(data_fb):
            predicted_target = process_trial_fbcca(trial, reference_signals, sub_bands)
            X.append(predicted_target)
            y.append(trial_idx % num_targets)  # Assuming trials are sequentially ordered

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier (SVM)
    clf = SVC(kernel='linear', C=1)
    clf.fit(np.array(X_train).reshape(-1, 1), y_train)

    # Evaluate
    y_pred = clf.predict(np.array(X_test).reshape(-1, 1))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()