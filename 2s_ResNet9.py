import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

##############################################################################
# 1) BASIC CONFIG AND FUNCTIONS
##############################################################################

# Selected electrodes: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
selected_electrodes = [48, 54, 55, 56, 57, 58, 61, 62, 63]  # 1-based
selected_indices = [i - 1 for i in selected_electrodes]     # 0-based

# We'll filter between 7..90 Hz
def preprocess_eeg(eeg_data, sampling_rate):
    """
    Band-pass filter [7,90] Hz with Chebyshev1, then standardize per-channel.
    Input shape: (channels, time)
    Output shape: (channels, time)
    """
    nyquist = sampling_rate / 2
    low, high = 7 / nyquist, 90 / nyquist
    b, a = signal.cheby1(N=4, rp=0.5, Wn=[low, high], btype='band')

    # Filter along 'time' axis=1
    filtered_eeg = signal.filtfilt(b, a, eeg_data, axis=1)

    # Standardize each channel
    mean_vals = np.mean(filtered_eeg, axis=1, keepdims=True)
    std_vals = np.std(filtered_eeg, axis=1, keepdims=True)
    normalized_eeg = (filtered_eeg - mean_vals) / (std_vals + 1e-12)

    return normalized_eeg

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """
    Basic residual block: 2 conv layers + skip connection
    """
    shortcut = x
    # If channel count differs, project shortcut
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides,
                          padding="same", kernel_regularizer=l2(0.001))(x)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=strides,
               padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(filters, kernel_size, strides=(1, 1),
               padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    from tensorflow.keras.layers import Add
    x = Add()([shortcut, x])
    x = Activation("elu")(x)
    return x

def calculate_itr(T, N, P):
    """
    Compute information transfer rate (bits/min).
    T = trial length in seconds
    N = number of classes
    P = accuracy in [0..1]
    """
    if P == 0 or P == 1:
        return 0
    return (60 / T) * (np.log2(N) + P*np.log2(P) + (1-P)*np.log2((1-P)/(N-1)))

##############################################################################
# 2) LOAD CONFIG AND PREPARE DATA
##############################################################################

with open("config.json", "r") as file:
    config = json.load(file)

data_dir = config["data_dir"]
sampling_rate = 250

# The 40 flicker frequencies (8..15.8 in steps of 0.2)
frequencies = np.arange(8, 16, 0.2)

X, y = [], []

# We have 5 total seconds (0.5..5.5s) => indices [125..1375] at 250 Hz
# We'll define multiple overlapping 2s windows inside that region.
# E.g. offsets = [0, 125, 250] => 3 windows: [125..625], [250..750], [375..875]
overlap_offsets = [0, 125, 250]   # each step = 0.5s

for mat_file in os.listdir(data_dir):
    if mat_file.endswith(".mat") and mat_file.startswith('S'):
        print(f"Processing file: {mat_file}")
        mat_contents = sio.loadmat(os.path.join(data_dir, mat_file))
        eeg_data = mat_contents['data']
        # eeg_data shape: (64, 1500, 40, 6)

        # Keep only the 9 desired electrodes
        eeg_data = eeg_data[selected_indices, :, :, :]

        for block_idx in range(eeg_data.shape[3]):  # 6 blocks
            for trial_idx in range(40):            # 40 stimuli
                # We'll create multiple overlapping segments from 0.5..3.5s
                for off in overlap_offsets:
                    start_index = 125 + off       # e.g. 125, 250, 375, etc.
                    end_index = start_index + 500 # 500 samples => 2s
                    # Make sure we don't go beyond 1375
                    if end_index > 1375:
                        break  # skip if we exceed available data

                    raw_trial = eeg_data[:, start_index:end_index, trial_idx, block_idx]
                    # shape: (9 channels, 500 time)

                    preprocessed_trial = preprocess_eeg(raw_trial, sampling_rate)
                    # => (9, 500)

                    # Expand dims => (9, 500, 1) for 2D CNN
                    preprocessed_trial = np.expand_dims(preprocessed_trial, axis=-1)

                    # Append
                    X.append(preprocessed_trial)
                    # label: trial_idx//6 lumps each freq set of 6 in the original code
                    # If you want distinct freq classes, you might do simply "trial_idx"
                    y.append(trial_idx // 6)

# Convert to arrays
X = np.array(X)  # shape (#samples, 9, 500, 1)
y = to_categorical(y, num_classes=len(frequencies))

##############################################################################
# 3) TRAIN-TEST SPLIT AND K-FOLD
##############################################################################

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.15, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies, fold_losses, fold_itrs = [], [], []
best_model = None
best_val_accuracy = 0

##############################################################################
# 4) BUILD & TRAIN THE MODEL (ResNet-like)
##############################################################################

for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Starting Fold {fold + 1} ---")
    fold_X_train, fold_X_test = X_train[train_idx], X_train[test_idx]
    fold_y_train, fold_y_test = y_train[train_idx], y_train[test_idx]

    # Input shape = (9, 500, 1)
    inputs = Input(shape=fold_X_train.shape[1:])

    # A basic CNN + residual blocks
    x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x)

    x = residual_block(x, filters=16)
    x = residual_block(x, filters=16)
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
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        fold_X_train, fold_y_train,
        epochs=30,
        batch_size=16,
        validation_data=(fold_X_test, fold_y_test),
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(fold_X_test, fold_y_test, verbose=0)
    fold_accuracies.append(test_acc)
    fold_losses.append(test_loss)

    # Now T=2 for each window. If you want "effective T" for an ensemble approach,
    # you might do something different, but let's keep T=2 for a single trial.
    itr = calculate_itr(2, len(frequencies), test_acc)
    fold_itrs.append(itr)

    if test_acc > best_val_accuracy:
        best_val_accuracy = test_acc
        best_model = model
        best_history = history

    print(f"Fold {fold + 1} - Accuracy: {test_acc:.3f}, Loss: {test_loss:.3f}, ITR: {itr:.2f} bits/minute")

# Final Cross-Validation Results
mean_acc = np.mean(fold_accuracies)
mean_loss = np.mean(fold_losses)
mean_itr = np.mean(fold_itrs)

print("\n--- Final Results (Cross-Val) ---")
print(f"Mean Accuracy: {mean_acc:.3f}")
print(f"Mean Loss: {mean_loss:.3f}")
print(f"Mean ITR: {mean_itr:.2f} bits/minute")

# Evaluate best model on hold-out set
eval_loss, eval_acc = best_model.evaluate(X_eval, y_eval, verbose=0)
eval_itr = calculate_itr(2, len(frequencies), eval_acc)

print("\n--- Evaluation on Hold-Out Set ---")
print(f"Accuracy: {eval_acc:.3f}, Loss: {eval_loss:.3f}, ITR: {eval_itr:.2f} bits/minute")

# Save metrics
output_file = "2s_raw_model_metrics.txt"
with open(output_file, "a") as f:
    f.write("\n--- Results: 2s Raw Data + Overlapping Windows + ResNet CV ---\n")
    f.write(f"Eval Accuracy: {eval_acc*100:.2f}%\n")
    f.write(f"Information Transfer Rate: {eval_itr:.2f} bits/minute\n")
    f.write(f"Eval Loss: {eval_loss:.3f}\n")
    f.write("-"*40 + "\n")

# Plot best fold training curves
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
