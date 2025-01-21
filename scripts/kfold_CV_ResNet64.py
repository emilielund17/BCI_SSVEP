import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Preprocessing function
def preprocess_eeg(eeg_data, sampling_rate):
    nyquist = sampling_rate / 2
    low, high = 7 / nyquist, 90 / nyquist
    b, a = signal.cheby1(N=4, rp=0.5, Wn=[low, high], btype='band')
    filtered_eeg = signal.filtfilt(b, a, eeg_data, axis=1)
    normalized_eeg = (filtered_eeg - np.mean(filtered_eeg, axis=1, keepdims=True)) / np.std(filtered_eeg, axis=1, keepdims=True)
    return normalized_eeg

def extract_features(eeg_data, sampling_rate, num_harmonics=3):
    fft_data = np.abs(np.fft.rfft(eeg_data, axis=1))
    freqs = np.fft.rfftfreq(eeg_data.shape[1], d=1 / sampling_rate)
    features = []
    for freq in np.arange(8, 16, 0.2):
        for h in range(1, num_harmonics + 1):
            idx = np.argmin(np.abs(freqs - (freq * h)))
            features.append(fft_data[:, idx])
    return np.array(features).T

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same", kernel_regularizer=l2(0.001))(x)
        shortcut = BatchNormalization()(shortcut)
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation("elu")(x)
    return x

def calculate_itr(T, N, P):
    if P == 0 or P == 1:
        return 0
    return (60 / T) * (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))

# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

data_dir = config["data_dir"]
sampling_rate = 250

# Preprocess and extract features
X, y = [], []
frequencies = np.arange(8, 16, 0.2)

for mat_file in os.listdir(data_dir):
    if mat_file.endswith(".mat") and mat_file.startswith('S'):
        print(f"Processing file: {mat_file}")
        mat_contents = sio.loadmat(os.path.join(data_dir, mat_file))
        eeg_data = mat_contents['data']
        for block_idx in range(eeg_data.shape[3]):
            for trial_idx in range(40):
                raw_trial = eeg_data[:, :, trial_idx, block_idx]
                preprocessed_trial = preprocess_eeg(raw_trial, sampling_rate)
                features = extract_features(preprocessed_trial, sampling_rate)
                X.append(features)
                y.append(trial_idx // 6)

X = np.array(X)
X = X[..., np.newaxis]
y = to_categorical(y, num_classes=len(frequencies))

# Split data
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15, random_state=42)

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies, fold_losses, fold_itrs = [], [], []
best_model = None
best_val_accuracy = 0

for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Starting Fold {fold + 1} ---")
    fold_X_train, fold_X_test = X_train[train_idx], X_train[test_idx]
    fold_y_train, fold_y_test = y_train[train_idx], y_train[test_idx]

    # Build model
    inputs = Input(shape=fold_X_train.shape[1:])
    x = Conv2D(16, (3, 3), padding="same", activation="elu", kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters=16)
    x = residual_block(x, filters=16)
    # x = MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="elu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(frequencies), activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    history = model.fit(fold_X_train, fold_y_train, 
                        epochs=20, 
                        batch_size=16, 
                        validation_data=(fold_X_test, fold_y_test), 
                        verbose=1)

    # Evaluate model
    test_loss, test_acc = model.evaluate(fold_X_test, fold_y_test)
    fold_accuracies.append(test_acc)
    fold_losses.append(test_loss)

    itr = calculate_itr(8, len(frequencies), test_acc)
    fold_itrs.append(itr)

    # Save the best model
    if test_acc > best_val_accuracy:
        best_val_accuracy = test_acc
        best_model = model
        best_history = history

    print(f"Fold {fold + 1} - Accuracy: {test_acc:.2f}, Loss: {test_loss:.2f}, ITR: {itr:.2f} bits/minute")

# Final Results
mean_accuracy = np.mean(fold_accuracies)
mean_loss = np.mean(fold_losses)
mean_itr = np.mean(fold_itrs)

print(f"\n--- Final Results ---")
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Mean Loss: {mean_loss:.2f}")
print(f"Mean ITR: {mean_itr:.2f} bits/minute")

# Evaluate the best model on hold-out validation set
eval_loss, eval_acc = best_model.evaluate(X_eval, y_eval)
eval_itr = calculate_itr(8, len(frequencies), eval_acc)
print(f"\n--- Evaluation on Hold-Out Set ---")
print(f"Accuracy: {eval_acc:.2f}, Loss: {eval_loss:.2f}, ITR: {eval_itr:.2f} bits/minute")

# Write metrics to a text file (append mode)
output_file = "new_model_metrics.txt"
with open(output_file, "a") as f:
    f.write(f"\n--- Results from Script: {'ResNet64 cross validation'} ---\n")
    f.write(f"Eval Accuracy: {eval_acc * 100:.2f}%\n")
    f.write(f"Information Transfer Rate (ITR): {eval_itr:.2f} bits/minute\n")
    #f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Eval loss:{eval_loss:.2f}\n")
    f.write("-" * 40 + "\n")

print(f"Metrics appended to {output_file}")

# Plot accuracy and loss curves for the best model
plt.figure(figsize=(12, 6))
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Best Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Best Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
