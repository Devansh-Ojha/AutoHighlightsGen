import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from preprocessing import audio_to_spectrogram, N_MELS, MAX_TIME_STEPS

DATA_DIR = "../data"

def cnn_build(input_shape):
    """
    Your VGG-style CNN Architecture
    """
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) # Output 0-1
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y = [], []
    categories = ["boring", "excitement"] # 0=Boring, 1=Excitement

    print("--- Loading Training Data ---")
    for label_idx, category in enumerate(categories):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder):
            print(f"Skipping {folder}, path not found.")
            continue
            
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                spec = audio_to_spectrogram(file_path=os.path.join(folder, fname))
                if spec is not None:
                    X.append(spec)
                    y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("ERROR: No data found in ../data folder.")
        return

    print(f"Training on {len(X)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = cnn_build(input_shape=(N_MELS, MAX_TIME_STEPS, 1))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    model.save("../cricket_model.h5")
    print(" Model saved to ../cricket_model.h5")

if __name__ == "__main__":
    train_model()