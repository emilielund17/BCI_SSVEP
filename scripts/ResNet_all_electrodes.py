import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
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
    for freq in np.arange(8, 16, 0.2):  # Stimulus frequencies
        for h in range(1, num_harmonics + 1):
            idx = np.argmin(np.abs(freqs - (freq * h)))
            features.append(fft_data[:, idx])

    return np.array(features).T

# Residual block function
def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    # Shortcut connection
    shortcut = x

    # Project the shortcut to match the number of filters if necessary
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same")(x)
        shortcut = BatchNormalization()(shortcut)

    # First convolution
    x = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    # Second convolution
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    # Add shortcut to the output
    x = Add()([shortcut, x])
    x = Activation("elu")(x)
    return x


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
X = X[..., np.newaxis]  # Add channel dimension for CNN

# Encode labels
y = to_categorical(y, num_classes=len(frequencies))

# Split data into train, test and validation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# Build ResNet Model
input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)

# Initial Conv Layer
x = Conv2D(16, (3, 3), padding="same", activation="elu")(inputs)
x = MaxPooling2D((2, 2))(x)

# Residual Blocks
x = residual_block(x, filters=16)
x = residual_block(x, filters=16)

# Down-sampling
x = MaxPooling2D((2, 2))(x)

# More Residual Blocks
x = residual_block(x, filters=32)
x = residual_block(x, filters=32)

# Flatten and Fully Connected Layers
x = Flatten()(x)
x = Dense(128, activation="elu")(x)
x = Dropout(0.5)(x)
outputs = Dense(len(frequencies), activation="softmax")(x)

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Get the script name
script_name = os.path.basename('ResNet all electrodes corrected eval 20 epochs')

# Start timing the training
start_time = time.time()

# Train the model and capture history
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# End timing
end_time = time.time()
training_time = end_time - start_time

# Function to calculate ITR
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

# Evaluate the model
test_loss, test_acc = model.evaluate(X_eval, y_eval)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Parameters for ITR
T = 8  # Trial duration in seconds (5 seconds stimulation + 0.5 seconds cue + 0.5 seconds rest+2 sec gaze shifting)
N = len(frequencies)  # Number of classes (e.g., 40)
P = test_acc  # Model's accuracy on test set

# Calculate ITR
itr = calculate_itr(T, N, P)
print(f"Information Transfer Rate (ITR): {itr:.2f} bits/minute")

# Write metrics to text file (append mode)
output_file = "model_metrics.txt"
with open(output_file, "a") as f:
    f.write(f"\n--- Results from Script: {script_name} ---\n")
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"Information Transfer Rate (ITR): {itr:.2f} bits/minute\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write("-" * 40 + "\n")

print(f"Metrics appended to {output_file}")

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