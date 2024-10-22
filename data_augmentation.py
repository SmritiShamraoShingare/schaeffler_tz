import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Define input and output folders
input_folder = r"D:\ng"  # Path to the folder containing original images
output_folder = r"D:\output_copies_ng" # Path to save augmented images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create an instance of ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=7,         # Rotate images by a maximum of 40 degrees
    width_shift_range=0.03,     # Shift images horizontally by a maximum of 20% of the width
    height_shift_range=0.03,    # Shift images vertically by a maximum of 20% of the height
    shear_range=0.03,           # Shear images by a maximum of 20%
    zoom_range=0.02,            # Zoom in on images by a maximum of 20%
    # horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill missing pixels after transformation using nearest pixels
)

# Function to augment and save images
def augment_images(input_folder, output_folder, save_prefix='aug', num_augmented=5, target_size=(16, 64)):
    for filename in os.listdir(input_folder):
        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path, target_size=target_size)  # Load and resize the image
            x = img_to_array(img)  # Convert the image to a NumPy array
            x = x.reshape((1,) + x.shape)  # Reshape for augmentation

            # Generate augmented images and save them
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder,
                                      save_prefix=save_prefix, save_format='jpeg'):
                i += 1
                if i >= num_augmented:
                    break  # Stop after generating the specified number of augmented images

# Example usage
augment_images(input_folder, output_folder, num_augmented=2, target_size=(32, 32))
