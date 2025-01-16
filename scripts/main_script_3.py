import numpy as np
import scipy.io as sio
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Preprocessing function
def preprocess_eeg(eeg_data, sampling_rate):
    """
    Preprocess the EEG data: filter and normalize.

    Parameters:
        eeg_data (np.ndarray): Raw EEG data with shape (channels, time_points).
        sampling_rate (int): Sampling rate of the EEG data.

    Returns:
        preprocessed_eeg (np.ndarray): Preprocessed EEG data.
    """
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
    """
    Extract features from EEG data using FFT for SSVEP classification.

    Parameters:
        eeg_data (np.ndarray): EEG data with shape (channels, time_points).
        sampling_rate (int): Sampling rate of the EEG data.
        num_harmonics (int): Number of harmonics to include.

    Returns:
        features (np.ndarray): Extracted features.
    """
    fft_data = np.abs(np.fft.rfft(eeg_data, axis=1))
    freqs = np.fft.rfftfreq(eeg_data.shape[1], d=1 / sampling_rate)

    # Select features at fundamental and harmonic frequencies
    features = []
    for freq in np.arange(8, 15.8, 0.2):  # Stimulus frequencies
        for h in range(1, num_harmonics + 1):
            idx = np.argmin(np.abs(freqs - (freq * h)))
            features.append(fft_data[:, idx])

    return np.array(features).T

# Load EEG Data
data_file = 'S01.mat'
data = sio.loadmat(data_file)
eeg_data = data['data']  # Shape: [64, 1500, 40, 6]
sampling_rate = 250

# Preprocess and extract features
X, y = [], []
frequencies = np.arange(8, 15.8, 0.2)

for subject_idx in range(eeg_data.shape[3]):
    for trial_idx in range(40):
        raw_trial = eeg_data[:, :, trial_idx, subject_idx]
        preprocessed_trial = preprocess_eeg(raw_trial, sampling_rate)
        features = extract_features(preprocessed_trial, sampling_rate)
        X.append(features)
        y.append(trial_idx // 6)  # Assuming 6 blocks per frequency

X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension for CNN

# Encode labels
y = to_categorical(y, num_classes=len(frequencies))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(frequencies), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")