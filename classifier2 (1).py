import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_size = (32, 32)
        img = cv2.resize(img, image_size)
        images.append(img)
    return images


# Load and label images
ng_images = load_images_from_folder(r"D:\output_copies_ng")
ok_images = load_images_from_folder(r"D:\output_copies_ok")
labels_ng = [0] * len(ng_images)
labels_ok = [1] * len(ok_images)

# Combine images and labels
X = np.array(ng_images + ok_images)
y = np.array(labels_ng + labels_ok)

# Reshape and normalize images
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = np.expand_dims(X_train, axis=-1) / 255.0  # Adding an extra dimension for grayscale images
X_test = np.expand_dims(X_test, axis=-1) / 255.0


# Build the model
def build_model(input_shape=(32, 32, 1), num_classes=2):
    model = Sequential()
    model.add(Conv2D(24, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(Conv2D(24, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(40, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(40, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(80, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(80, (3, 3), activation="relu", padding="same"))

    model.add(GlobalAveragePooling2D())

    # Change to a single output with sigmoid for binary classification
    model.add(Dense(1, activation="sigmoid"))

    return model


# Custom callback to save the model every fourth epoch
class SaveModelEveryFourthEpoch(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 4 == 0:
            super().on_epoch_end(epoch, logs)


# Check if a model exists, else build a new one
model_filepath = 'FAG_classifier_model_part_1.keras'
if os.path.exists(model_filepath):
    model = load_model(model_filepath)
else:
    model = build_model()

# Compile the model with a custom learning rate
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Custom callback
checkpoint_filepath = 'FAG_classifier_model_part_1.keras'
model_checkpoint = SaveModelEveryFourthEpoch(checkpoint_filepath, save_weights_only=False, save_best_only=True,
                                             monitor='val_accuracy', mode='max', verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[model_checkpoint])

# Save the model after training
model.save(model_filepath)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
