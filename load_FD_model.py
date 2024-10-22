import os
from segmentation_models.metrics import iou_score
import numpy as np
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

            # Calculate cropping coordinates to crop the center of the image
            # crop_x1 = 353 ### stn 1 cam A
            # crop_y1 = 20
            # crop_x2 = 897
            # crop_y2 = 572

            crop_x1 = 166 ### stn 2 cam A
            crop_y1 = 197
            crop_x2 = 470
            crop_y2 = 405

            cropped_img_array = img_array[crop_y1:crop_y2, crop_x1:crop_x2, :]

            resized_img = cv2.resize(cropped_img_array, (160, 96))

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

        cv2.imshow(f'prediction image', prediction_image_rgb)
        cv2.waitKey(0)

        img = array_to_img(prediction_image_rgb)  # Converts the array to an image

        # Save the prediction image
        output_file = os.path.join(output_folder, f"{filename}_prediction.png")
        img.save(output_file)

        output_files.append(output_file)

    return output_files


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


       else:
           black_mask = (fd_image == 0)  # Create mask for black pixels (shape: (height, width))
       print(black_mask.shape, 'shape')

       img_array[black_mask] = [0,0,0]  # Set black areas in the output image

       # Save the modified output image to the final folder
       final_output_filename = os.path.join(final_folder, f"final_{i}.png")
       final_output_image = img_array.astype(np.uint8)
       cv2.imwrite(final_output_filename, final_output_image)

   print(f"Processed images saved to {final_folder}")


def main():

        model = load_model(model_path, custom_objects={'iou_score': iou_score})
        print(f"Model loaded successfully from {model_path}")

        # Load and preprocess images from the input folder
        images, filenames = load_images(input_folder)

        # Ensure images have the correct shape
        print(f"Loaded images shape: {images.shape}")

        threshold = 0.5

        # Run the model to make predictions
        predictions = model.predict(images)

        filtered_predictions = (predictions > threshold).astype(np.uint8)  # Convert boolean to uint8 for binary mask

        # save_predictions(filtered_predictions, filenames, output_folder)
        output_file = save_predictions_new(filtered_predictions, filenames, output_folder)

        apply_fd(input_folder, output_file, final_folder)

if __name__ == "__main__":
    main()
