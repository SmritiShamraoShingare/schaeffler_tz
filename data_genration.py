import cv2
import numpy as np
import random
import os

# Define the input and output directories
input_dir = r"D:\output_copies"
output_dir = r"D:\ng"

# Check if the output directory exists, and if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist


# Define a function to find a random pixel not close to black
def get_random_non_black_pixel(img, threshold=50):
    while True:
        # Randomly select a pixel coordinate
        x = random.randint(0, img.shape[1] - 1)
        y = random.randint(0, img.shape[0] - 1)

        # Get the pixel color (BGR format)
        pixel = img[y, x]

        # Check if the pixel is not close to black (i.e., all color channels are above the threshold)
        if np.all(pixel > threshold):
            return (x, y), pixel


# Function to create random patches centered around the image center
def create_random_patches(img, pixel_color, patch_size=(20, 20), num_patches=3):
    # Get the center of the image
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    # Define the range within which the patch's top-left corner should be centered around the image center
    offset_x = img.shape[1] // 4  # Range limit around center for x
    offset_y = img.shape[0] // 4  # Range limit around center for y

    for _ in range(num_patches):
        # Randomly choose the top-left corner for the patch within the range of the center
        x = random.randint(center_x - offset_x, center_x + offset_x - patch_size[0])
        y = random.randint(center_y - offset_y, center_y + offset_y - patch_size[1])

        # Define the patch region
        patch = img[y:y + patch_size[1], x:x + patch_size[0]]

        # Fill the patch with the selected pixel color
        patch[:] = pixel_color


# Process each image in the input folder
for file_name in os.listdir(input_dir):
    # Check if the file is an image (e.g., ends with .png, .jpg, .jpeg, .tif, etc.)
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        img_path = os.path.join(input_dir, file_name)

        # Load the image
        image = cv2.imread(img_path)

        # Skip if the image could not be loaded
        if image is None:
            print(f"Could not load image: {file_name}")
            continue

        # Create 10 copies for each input image with random patches
        for i in range(5):
            # Clone the original image to avoid modifying it directly
            modified_image = image.copy()

            # Find a random non-black pixel
            pixel_coord, pixel_color = get_random_non_black_pixel(modified_image)

            # Create random patches on the cloned image, centered around the image's center
            create_random_patches(modified_image, pixel_color, patch_size=(4, 4), num_patches=6)

            # Save the modified image with a unique file name
            output_path = os.path.join(output_dir, f"{file_name}_modified_{i + 1}.png")
            cv2.imwrite(output_path, modified_image)

print(f"All images have been processed and saved to: {output_dir}")
