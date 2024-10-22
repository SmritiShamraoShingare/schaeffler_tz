import os
from segmentation_models.metrics import iou_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Hardcoded paths
# input_folder = r"C:\Users\SHUBHAM\Desktop\stn1_seal"  # Path to input images
input_folder = r"C:\Users\SHUBHAM\Documents\stn2_cam_A"  # Path to input images
output_folder = r"D:\code_data_crops_768_and_384\output_348\testing_output"  # Path to save predictions
final_folder = r"D:\code_data_crops_768_and_384\output_348\testing_final___"
# model_path = r"C:\Users\SHUBHAM\Desktop\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1\Schaeffler_6201.C.H305_A_stn1_cam_A_Top_viewPort1_model_from_best.h5" # Path to the saved model
model_path = r"C:\Users\SHUBHAM\Documents\Schaeffler_6201.C.H305_A_stn2_cam_A_BoreCenter_viewPort1\Schaeffler_6201.C.H305_A_stn2_cam_A_BoreCenter_viewPort1_model_from_best.h5" # Path to the saved model


# Function to load and crop images from the input folder
def load_images(input_folder, target_size):
    images = []
    filenames = []

    target_height, target_width = target_size

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Add any additional formats you need
            file_path = os.path.join(input_folder, filename)

            # Load the image with its original size
            img = load_img(file_path, color_mode='rgb')

            # Convert to numpy array
            img_array = img_to_array(img)
            original_height, original_width, _ = img_array.shape

            # Calculate cropping coordinates to crop the center of the image
            # crop_x1 = 353 ### stn 1 cam A
            # crop_y1 = 20
            # crop_x2 = 897
            # crop_y2 = 572

            crop_x1 = 166 ### stn 2 cam A
            crop_y1 = 197
            crop_x2 = 470
            crop_y2 = 405


            # Crop the image
            cropped_img_array = img_array[crop_y1:crop_y2, crop_x1:crop_x2, :]
            # plt.imshow(cropped_img_array)
            # plt.show()


            # Resize the image to 190x190
            resized_img = cv2.resize(cropped_img_array, (160, 96))


            # Append the cropped image to the list
            images.append(resized_img)

            # Save the filename for later use
            filenames.append(filename)

    images = np.array(images)
    return images, filenames

# def save_predictions_new(predictions, filenames, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#
#     for i, filename in enumerate(filenames):
#         # Take the first channel of the prediction image
#         prediction_image = predictions[i, :, :, 0]  # Shape (height, width)
#
#         # Ensure the prediction image is in the correct format for saving as grayscale
#         prediction_image = np.expand_dims(prediction_image, axis=-1)  # Shape becomes (height, width, 1)
#
#         # Convert the array to an image
#         img = array_to_img(prediction_image)  # 'L' mode for grayscale
#
#         # Save the prediction image as grayscale
#         output_file = os.path.join(output_folder, f"{filename}_prediction.png")
#         img.save(output_file)
# # Main function to load the model, process input, and save output
#     return output_file

def save_predictions_new(predictions, filenames, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    output_files = []  # To store paths to the saved files

    for i, filename in enumerate(filenames):
        for j in range(0,2):
            # Take the first channel of the prediction image
            prediction_image = predictions[i, :, :, j]  # Shape (height, width)

            # Ensure the prediction image is in the correct format for saving as grayscale
            prediction_image = np.expand_dims(prediction_image, axis=-1)  # Shape becomes (height, width, 1)

            # Convert to 3-channel grayscale image (optional if your model expects 3 channels)
            prediction_image_rgb = np.repeat(prediction_image, 3, axis=-1)  # Shape becomes (height, width, 3)
            #
            # cv2.imshow(f'prediction image', prediction_image_rgb)
            # cv2.waitKey(0)

            # Convert the array to an image
            img = array_to_img(prediction_image_rgb)  # Converts the array to an image

            # Save the prediction image
            output_file = os.path.join(output_folder, f"{filename}_prediction_{j}.png")
            img.save(output_file)

            # Append to the list of saved filesq
            output_files.append(output_file)

    return output_files  # Return the list of saved files


def apply_fd(input_folder, output_files, final_folder):
   os.makedirs(final_folder, exist_ok=True)
   output_images = []
   original_image = []
   # original_image_array = []
   # target_size = (704, 736)
   # original_image_array ,_=  load_images(input_folder, target_size)

   for filename in os.listdir(input_folder):
       if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Supported formats
           file_path = os.path.join(input_folder, filename)
           img = cv2.imread(file_path)  # Read the image using OpenCV
           crop_x1 = 353
           crop_y1 = 20
           crop_x2 = 897
           crop_y2 = 572

           # Crop the image
           cropped_img_array = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

           # Resize the image to 190x190
           resized_img = cv2.resize(cropped_img_array, (160, 96))

       original_image.append(resized_img)

   original_image_array = np.array(original_image)

   # Load images from the original folder
   for filename in os.listdir(output_folder):
       if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Supported formats
           file_path = os.path.join(output_folder, filename)
           img = cv2.imread(file_path)  # Read the image using OpenCV
           img_resized = cv2.resize(img, (160, 96))  # Resize to 1920x1080
           output_images.append(img_resized)

   output_images_array = np.array(output_images)

   for i, img_array in enumerate(original_image_array):

       # Create a mask where the input image is black (all channels are 0)
       fd_image = output_images_array[i]
       print(img_array.shape, 'shape')
       if img_array.ndim == 3:  # Color image (height, width, channels)
           black_mask = np.all(fd_image == 0, axis=-1)  # Shape: (height, width)


       else:  # Grayscale image (height, width)
           black_mask = (fd_image == 0)  # Create mask for black pixels (shape: (height, width))
       print(black_mask.shape, 'shape')

       # Set the pixels in the output image where the input image is black to black
       img_array[black_mask] = [0,0,0]  # Set black areas in the output image

       # Save the modified output image to the final folder
       final_output_filename = os.path.join(final_folder, f"final_{i}.png")  # Modify as needed
       final_output_image = img_array.astype(np.uint8)  # Ensure the output image is in uint8 format
       cv2.imwrite(final_output_filename, final_output_image)

   print(f"Processed images saved to {final_folder}")


def main():

        # Load the pre-trained model
        model = load_model(model_path, custom_objects={'iou_score': iou_score})
        print(f"Model loaded successfully from {model_path}")

        # Define a fixed target size (height, width)
        target_size = (704, 736)  # Use a fixed size suitable for your model
        print(f"Target size for images: {target_size}")

        # Load and preprocess images from the input folder
        images, filenames = load_images(input_folder, target_size)

        # Ensure images have the correct shape
        print(f"Loaded images shape: {images.shape}")

        threshold = 0.5 # Set your threshold value here

        # Run the model to make predictions
        predictions = model.predict(images)


        # plt.imshow(single_image[:, :, 1])
        # plt.show()

        print(predictions, 'predictions ')

        # if you want to keep it binary (0 or 1):
        filtered_predictions = (predictions > threshold).astype(np.uint8)  # Convert boolean to uint8 for binary mask

        feature0 = filtered_predictions[0]  # Extract the first image
        feature1 = filtered_predictions[1]  # Extract the first image

        plt.imshow(feature0[:, :, 1])
        plt.show()


        # save_predictions(filtered_predictions, filenames, output_folder)
        output_file = save_predictions_new(filtered_predictions, filenames, output_folder)

        apply_fd(input_folder, output_file, final_folder)

# Run the main function
if __name__ == "__main__":
    main()
