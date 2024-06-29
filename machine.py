import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 16000
DURATION = 10  # Duration of each audio clip in seconds
NUM_CLASSES = 3  # Number of hive states (e.g., healthy, queenless, sick)
LOG_DIR = "logs/"
DATA_DIR = 'hive_audio_data'

# Function to preprocess audio data
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    return mfccs

# Load data and labels
def load_data(data_dir):
    data = []
    labels = []
    classes = os.listdir(data_dir)
    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            mfccs = preprocess_audio(file_path)
            data.append(mfccs)
            labels.append(i)
    return np.array(data), np.array(labels)

# Split data into training and validation sets
def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Build deep learning model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function for training the model
def train_model(data, labels):
    input_shape = data[0].shape
    data_train, data_val, labels_train, labels_val = split_data(data, labels)

    # Convert labels to one-hot encoding
    labels_train = to_categorical(labels_train, num_classes=NUM_CLASSES)
    labels_val = to_categorical(labels_val, num_classes=NUM_CLASSES)

    # Build and train the model
    model = build_model(input_shape)
    callbacks = [
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        TensorBoard(log_dir=LOG_DIR)
    ]
    history = model.fit(data_train, labels_train, epochs=20, batch_size=32, validation_data=(data_val, labels_val), callbacks=callbacks)

    # Evaluate the model
    loss, accuracy = model.evaluate(data_val, labels_val)
    print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

    # Save the trained model
    model.save('hive_state_detection_model.h5')
    return history

# Function for making predictions
def predict_hive_state(model, audio_file):
    # Preprocess the audio file
    mfccs = preprocess_audio(audio_file)

    # Perform inference
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

# Hyperparameter tuning
# Hyperparameter tuning
def tune_hyperparameters(data, labels):
    input_shape = data[0].shape
    data_train, data_val, labels_train, labels_val = split_data(data, labels)

    def objective(params):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(params['dense_units'], activation='relu'),
            Dropout(params['dropout']),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(data_train, labels_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(data_val, labels_val), verbose=0)
        loss, accuracy = model.evaluate(data_val, labels_val)
        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    space = {
        'dense_units': hp.choice('dense_units', [256, 512]),
        'dropout': hp.uniform('dropout', 0.3, 0.7),
        'epochs': hp.choice('epochs', [10, 20]),
        'batch_size': hp.choice('batch_size', [32, 64])
    }

    trials = Trials()
    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    best_model = trials.best_trial['result']['model']
    return best_params, best_model

# Data augmentation
def augment_data(data, labels):
    augmented_data = []
    augmented_labels = []
    for i in range(len(data)):
        augmented_data.append(data[i])
        augmented_labels.append(labels[i])
        augmented_data.append(np.flip(data[i], axis=1))  # Horizontal flip
        augmented_labels.append(labels[i])
        augmented_data.append(np.rot90(data[i], k=1, axes=(0, 1)))  # Rotate 90 degrees
        augmented_labels.append(labels[i])
    return np.array(augmented_data), np.array(augmented_labels)

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data, labels = load_data(DATA_DIR)

    # Hyperparameter tuning
    best_params, best_model = tune_hyperparameters(data, labels)
    print("Best Hyperparameters:", best_params)

    # Data augmentation
    augmented_data, augmented_labels = augment_data(data, labels)

    # Training with augmented data
    history = train_model(augmented_data, augmented_labels)

    # Plot training history
    plot_training_history(history)

    # Evaluate the best model
    # Split data into training and validation sets
    data_train, data_val, labels_train, labels_val = split_data(data, labels)

    best_loss, best_accuracy = best_model.evaluate(data_val, labels_val)
    print(f'Best Model Validation Loss: {best_loss}, Validation Accuracy: {best_accuracy}')

    # Save the best model
    best_model.save('best_hive_state_detection_model.h5')

    # Load the best model for prediction
    best_model = load_model('best_hive_state_detection_model.h5')
    predicted_class = predict_hive_state(best_model, 'test_audio.wav')
    print(f'Predicted hive state: {predicted_class}')
