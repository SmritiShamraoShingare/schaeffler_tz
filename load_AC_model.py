import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil

# Load the pre-trained model
model = load_model(r"D:\smriti (copy)\FAG_classifier_model_part_1.keras")

# Define output folders for different classes
ok = r"D:\code_data_crops_768_and_384\output_348\FAG\ok"  # Path for saving Class 1 predictions
ng = r"D:\code_data_crops_768_and_384\output_348\FAG\ng"  # Path for saving Class 2 predictions

# Create output directories if they don't exist
os.makedirs(ok, exist_ok=True)
os.makedirs(ng, exist_ok=True)

# Function to load and preprocess the input image
def preprocess_image(img_path, target_size=(32, 32)):
    # Load the image in grayscale mode
    img = image.load_img(img_path, color_mode='grayscale', target_size=target_size)

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's expected input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image to the [0, 1] range (if the model expects this)
    img_array = img_array / 255.0

    return img_array

# Function to make a prediction on an image and save it to the corresponding class folder
def predict_and_save_image(img_path):
    preprocessed_image = preprocess_image(img_path)

    prediction = model.predict(preprocessed_image)

    # Determine the predicted class and save to the corresponding output folder
    if prediction[0] > 0.5:
        # Predicted as Class 1
        output_path = os.path.join(ok, os.path.basename(img_path))
        shutil.copy(img_path, output_path)  # Copy the image to the Class 1 folder
        print(f"{os.path.basename(img_path)} predicted as Class 1 and saved to {ok}")
    else:
        # Predicted as Class 0
        output_path = os.path.join(ng, os.path.basename(img_path))
        shutil.copy(img_path, output_path)  # Copy the image to the Class 2 folder
        print(f"{os.path.basename(img_path)} predicted as Class 0 and saved to {ng}")

# Function to predict for all images in a folder
def predict_folder_and_save(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            predict_and_save_image(img_path)

# Example usage
input_folder = r"D:\code_data_crops_768_and_384\output_348\FAG\testing"  # Path to input folder containing images
predict_folder_and_save(input_folder)
