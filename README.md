Gender Classification using TensorFlow
This project implements a gender classification model using TensorFlow and transfer learning with MobileNetV2. The model is trained on a dataset of facial images to predict the gender of the person in the image.

Overview
The gender classification model is built using the following steps:

Data Preparation:
The dataset consists of images of faces, categorized by gender (male or female).
Data augmentation techniques such as shear, zoom, and horizontal flip are applied to increase the diversity of the training data.
Model Architecture:
Transfer learning is used with the MobileNetV2 pre-trained model as the base.
Custom classification layers are added on top of the MobileNetV2 base to adapt it to the gender classification task.
The model is compiled with binary cross-entropy loss and Adam optimizer with a lower learning rate.
Training:
The model is trained on the prepared dataset, with early stopping and learning rate reduction callbacks to prevent overfitting.
Training progress is monitored using both training and validation datasets.
Evaluation:
Model performance is evaluated on a separate validation dataset using accuracy as the metric.
Deployment:
The trained model is saved for future use and deployment in other applications.
Usage
To use this project:

Clone the repository to your local machine:

git clone https://github.com/yajjuhemanth/gender-classification-by-image.git
Install the required dependencies:

pip install -r requirements.txt
Run the gender_classification.py script to train the model:

python gender_classification.py
Once trained, you can use the saved model (gender_classification_model_v2.h5) for gender classification tasks in other applications.
