import os

from matplotlib import pyplot as plt
from segmentation_models.metrics import iou_score
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Hardcoded paths
# input_folder = r"C:\Users\SHUBHAM\Desktop\stn1_seal"  # Path to input images
input_folder = r"D:\code_data_crops_768_and_384\testing_straight"  # Path to input images
output_folder = r"D:\code_data_crops_768_and_384\output_348\1_testing_output"  # Path to save predictions
final_folder = r"D:\code_data_crops_768_and_384\output_348\testing_final___"
# model_path = r"C:\Users\SHUBHAM\Desktop\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1_model_from_best.h5" # Path to the saved model
model_path = r"C:\Users\SHUBHAM\Downloads\schaeffler_variant_MM10B2_stn4_cam_A_viewport10\schaeffler_variant_MM10B2_stn4_cam_A_viewport10_model_from_best.h5" # Path to the saved model


# Function to load and crop images from the input folder
def load_images(input_folder):
    images = []
    filenames = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Add any additional formats you need
            file_path = os.path.join(input_folder, filename)

            # Load the image with its original size
            img = load_img(file_path, color_mode='rgb')

            # Convert to numpy array
            img_array = img_to_array(img)
            original_height, original_width, _ = img_array.shape
            crop_x1 = 0 ### stn 2 cam A
            crop_y1 = 0
            crop_x2 = 2086
            crop_y2 = 30

            cropped_img_array = img_array[crop_y1:crop_y2, crop_x1:crop_x2, :]

            resized_img = cv2.resize(cropped_img_array, (2080, 32))

            images.append(resized_img)

            filenames.append(filename)

    images = np.array(images)
    return images, filenames


def save_predictions_new(predictions, filenames, output_folder):


    os.makedirs(output_folder, exist_ok=True)

    output_files = []  # To store paths to the saved files

    for i, filename in enumerate(filenames):

        prediction_image = predictions[i, :, :, 0]  # Shape (height, width)


        prediction_image = np.expand_dims(prediction_image, axis=-1)  # Shape becomes (height, width, 1)

        prediction_image_rgb = np.repeat(prediction_image, 3, axis=-1)  # Shape becomes (height, width, 3)

        img = array_to_img(prediction_image_rgb)  # Converts the array to an image

        # Save the prediction image
        output_file = os.path.join(output_folder, f"{filename}_prediction.png")
        img.save(output_file)

        output_files.append(output_file)

def apply_fd(input_folder, final_folder, images_array):

   # print('in apply fd output file shape', output_files[0].shape)
   os.makedirs(final_folder, exist_ok=True)
   output_images = []

   # Load images from the original folder
   for filename in os.listdir(output_folder):
       if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Supported formats
           file_path = os.path.join(output_folder, filename)
           img = cv2.imread(file_path)  # Read the image using OpenCV

           img_resized = cv2.resize(img, (2080, 32))  # Resize to 1920x1080

           output_images.append(img_resized)

   output_images_array = np.array(output_images)

   for i, img_array in enumerate(images_array):

       # Create a mask where the input image is black (all channels are 0)
       fd_image = output_images_array[i]

       if img_array.ndim == 3:  # Color image (height, width, channels)
           black_mask = np.all(fd_image == 0, axis=-1)  # Shape: (height, width)
       else:
           black_mask = (fd_image == 0)

       img_array[black_mask] = [0,0,0]

       # Save the modified output image to the final folder
       final_output_filename = os.path.join(final_folder, f"final_{i}.png")
       final_output_image = img_array.astype(np.uint8)
       cv2.imwrite(final_output_filename, final_output_image)

   print(f"Processed images saved to {final_folder}")

# def crop_final_input():



def main():
        images_array = []
        model = load_model(model_path, custom_objects={'iou_score': iou_score})
        print(f"Model loaded successfully from {model_path}")

        # Load and preprocess images from the input folder
        images, filenames = load_images(input_folder)

        for i, img in enumerate(images):
            images_array.append(img)

        #
            img = np.expand_dims(img, axis=0)  # Add batch dimension
        #
            predictions_image = model.predict(img)
            print('shape', predictions_image[0].shape)

            # plt.imshow(predictions_image[0])
            # plt.show()

            thresholded = (predictions_image[0] > 0.9).astype(np.uint8) * 255  # Values > 0.9 will be set to 255, others to 0

            # Find the total number of white pixels (255)
            white_pixels = np.sum(thresholded == 255)


        # Optionally, find white pixels along specific axes
            white_pixels_in_columns = np.sum(thresholded == 255, axis=0)  # Sum along each column
            white_pixels_in_rows = np.sum(thresholded == 255, axis=1)  # Sum along each row

            # Print the total count of white pixels
            print("Total white pixels:", white_pixels)


            if np.any(white_pixels < 600):
                print('not matched')

                cropped = img[:, :, :50]
                remaining_image = img[:, :, 50:]

                concatenated_image = np.concatenate((remaining_image, cropped), axis=2)
                final_image = concatenated_image.astype(np.uint8)


                concatenated_image = np.squeeze(concatenated_image)

                # images_array.append(concatenated_image)
                images[i] = concatenated_image


        images_array = np.array(images_array)

        # predictions = model.predict(images_array)
        predictions = model.predict(images_array)

        save_predictions_new(predictions, filenames, output_folder)

        apply_fd(input_folder, final_folder, images_array)

if __name__ == "__main__":
    main()
