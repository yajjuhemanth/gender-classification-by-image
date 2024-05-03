import os
import cv2
import numpy as np
import tensorflow as tf

# Create output directories for male and female images
output_male_dir = 'D:/photos/magic/male_images'
output_female_dir = 'D:/photos/magic/female_images'
os.makedirs(output_male_dir, exist_ok=True)
os.makedirs(output_female_dir, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('gender_classification_model.h5')

# Define data paths
input_data_dir = 'D:\photos\magic\IFHE\BTECH\Batch 22'

# Define image parameters
img_width, img_height = 224, 224

# Iterate over images in the input directory
for file_name in os.listdir(input_data_dir):
    file_path = os.path.join(input_data_dir, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):
        # Load and preprocess the image
        image = cv2.imread(file_path)
        image = cv2.resize(image, (img_width, img_height))
        image = np.expand_dims(image, axis=0) / 255.0

        # Predict the gender of the image
        prediction = model.predict(image)[0, 0]

        # Determine the gender label
        gender = 'male' if prediction >= 0.5 else 'female'

        # Save the image in the corresponding output directory
        output_dir = output_male_dir if gender == 'male' else output_female_dir
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, cv2.imread(file_path))

        print(f"Image '{file_name}' separated as {gender} and saved to {output_dir}")
