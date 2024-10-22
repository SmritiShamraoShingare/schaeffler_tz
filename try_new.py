import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from segmentation_models.metrics import iou_score

input_folder = r"D:\code_data_crops_768_and_384\testing_straight"  # Path to input images
output_folder = r"D:\code_data_crops_768_and_384\output_348\1_testing_output"  # Path to save predictions
final_folder = r"D:\code_data_crops_768_and_384\output_348\testing_final___"
# model_path = r"C:\Users\SHUBHAM\Desktop\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1_model_from_best.h5" # Path to the saved model
model_path = r"C:\Users\SHUBHAM\Downloads\schaeffler_variant_MM10B2_stn4_cam_A_viewport10\schaeffler_variant_MM10B2_stn4_cam_A_viewport10_model_from_best.h5" # Path to the saved model


# Function to load and crop images from the input folder
def load_image(file_path):
    # Load and convert to numpy array
    img = load_img(file_path, color_mode='rgb')
    img_array = img_to_array(img)

    # Crop and resize the image
    cropped_img_array = img_array[0:30, 0:2086, :]
    resized_img = cv2.resize(cropped_img_array, (2080, 32))
    return resized_img

# Function to predict with an option for concatenated images
def predict_and_adjust(model, img_array):
    img_input = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_input)[0]

    # Threshold and find white pixels
    thresholded = (prediction > 0.9).astype(np.uint8) * 255
    white_pixels = np.sum(thresholded == 255)

    if white_pixels < 600:
        print("Not matched, applying adjustment.")
        cropped = img_array[:, :50]
        remaining_image = img_array[:, 50:]
        img_array = np.concatenate((remaining_image, cropped), axis=1)

        # Re-predict on the concatenated image
        img_input = np.expand_dims(img_array, axis=0)  # Add batch dimension again
        prediction = model.predict(img_input)[0]

    return prediction, img_array


def apply_fd(fd_image, final_image):

    plt.imshow(fd_image)
    plt.show()

    cv2.imshow(f'image', final_image)
    cv2.waitKey(0)

    if fd_image.ndim == 3:
        black_mask = np.all(fd_image == 0, axis=-1)
    else:
        black_mask = (fd_image > 0.5)

    final_image[black_mask] = [0, 0, 0]

    return final_image

# Main function to process images and apply the model
def main():
    model = load_model(model_path, custom_objects={'iou_score': iou_score})
    print(f"Model loaded successfully from {model_path}")

    os.makedirs(final_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            file_path = os.path.join(input_folder, filename)

            # Load and preprocess images
            img_array = load_image(file_path)

            prediction, adjusted_image = predict_and_adjust(model, img_array)

            final_image = apply_fd(prediction, adjusted_image)

            final_output_file = os.path.join(final_folder, f"final_{filename}")
            final_image = final_image.astype(np.uint8)  # Ensure the output is in uint8 format
            cv2.imwrite(final_output_file, final_image)
            print(f"Saved final output for {filename} to {final_output_file}")

if __name__ == "__main__":
    main()
