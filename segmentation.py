import os
import tensorflow as tf
import cv2
import numpy as np
import gdown

# Define global variables
# SIZE_X and SIZE_Y are the dimensions of the input image for the segmentation model
SIZE_X = 256 
SIZE_Y = 256
CLASS_NUM = 2  # Tumor and non-tumor (binary segmentation)

# Define the dice coefficient function
def dice_coef(y_true, y_pred, smooth=1.0):
    dice = 0
    for i in range(CLASS_NUM):
        y_true_f = tf.keras.backend.flatten(y_true[:, :, :, i])
        y_pred_f = tf.keras.backend.flatten(y_pred[:, :, :, i])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice += (2. * intersection + smooth) / (union + smooth)
    dice /= CLASS_NUM
    return dice
    
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (SIZE_Y, SIZE_X))
    img_gray = cv2.merge([img_gray, img_gray, img_gray])  # Convert grayscale to 3-channel image
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    gaussian = cv2.GaussianBlur(laplacian, (5, 5), 0)  # Apply Laplacian of Gaussian filter
    return gaussian

def predict_segment(preprocessed_img, model):
    prediction = model.predict(preprocessed_img)
    prediction = np.argmax(prediction, axis=3)[0,:,:]
    return prediction