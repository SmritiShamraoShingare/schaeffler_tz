import os
from segmentation_models.metrics import iou_score
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Hardcoded paths
input_folder = r"D:\code_data_crops_768_and_384\testing_straight"  # Path to input images
output_folder = r"D:\code_data_crops_768_and_384\output_348\1_testing_output"  # Path to save predictions
final_folder = r"D:\code_data_crops_768_and_384\output_348\testing_final___"
# model_path = r"C:\Users\SHUBHAM\Desktop\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1_model_from_best.h5" # Path to the saved model
model_path = r"C:\Users\SHUBHAM\Downloads\schaeffler_variant_MM10B2_stn4_cam_A_viewport10\schaeffler_variant_MM10B2_stn4_cam_A_viewport10_model_from_best.h5" # Path to the saved model


# Function to process each image one by one
def process_image(filename, model):
    file_path = os.path.join(input_folder, filename)

    # Load the image
    img = load_img(file_path, color_mode='rgb')
    img_array = img_to_array(img)

    # Crop and resize
    crop_x1 = 0  ### stn 2 cam A
    crop_y1 = 0
    crop_x2 = 2086
    crop_y2 = 30
    # For station 2 cam A
    cropped_img_array = img_array[crop_y1:crop_y2, crop_x1:crop_x2, :]
    resized_img = cv2.resize(cropped_img_array, (2080, 32))

    # Expand dimensions to fit model input (batch of 1)
    input_img = np.expand_dims(resized_img, axis=0)

    # Predict
    prediction = model.predict(input_img)[0]  # Get the first prediction from batch

    # Convert prediction to binary mask
    threshold = 0.5
    filtered_prediction = (prediction > threshold).astype(np.uint8)

    # Save prediction
    save_prediction(filtered_prediction, filename, output_folder)

    # Apply fault detection (fd)
    apply_fd(filename, output_folder, final_folder)


# Function to save predictions
def save_prediction(prediction, filename, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    prediction_image = np.expand_dims(prediction[:, :, 0], axis=-1)  # (height, width, 1)
    prediction_image_rgb = np.repeat(prediction_image, 3, axis=-1)  # (height, width, 3)

    img = array_to_img(prediction_image_rgb)
    output_file = os.path.join(output_folder, f"{filename}_prediction.png")
    img.save(output_file)


# Function to apply fault detection
def apply_fd(filename, output_folder, final_folder):
    os.makedirs(final_folder, exist_ok=True)

    # Load the original image
    original_file_path = os.path.join(input_folder, filename)
    original_img = cv2.imread(original_file_path)

    # Crop and resize the original image
    crop_x1 = 0  ### stn 2 cam A
    crop_y1 = 0
    crop_x2 = 2086
    crop_y2 = 30


    cropped_img_array = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    resized_original_img = cv2.resize(cropped_img_array, (2080, 32))

    # Load the prediction image
    prediction_file_path = os.path.join(output_folder, f"{filename}_prediction.png")
    prediction_img = cv2.imread(prediction_file_path)

    # Create a mask for black areas
    black_mask = np.all(prediction_img == 0, axis=-1)
    resized_original_img[black_mask] = [0, 0, 0]

    # Save the final image
    final_output_filename = os.path.join(final_folder, f"final_{filename}")
    cv2.imwrite(final_output_filename, resized_original_img)


def main():
    model = load_model(model_path, custom_objects={'iou_score': iou_score})
    print(f"Model loaded successfully from {model_path}")

    # Process each image one by one
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Supported formats
            process_image(filename, model)


if __name__ == "__main__":
    main()
