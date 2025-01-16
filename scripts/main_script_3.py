import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.utils import to_categorical

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
    for freq in np.arange(8, 15.8, 0.2):  # Stimulus frequencies
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
frequencies = np.arange(8, 15.8, 0.2)

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

# Train the model and capture history
history=model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

