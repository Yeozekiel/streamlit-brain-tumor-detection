import numpy as np
import tensorflow as tf

def predict_class(input_image, model):
    input_image = np.array(input_image)
    input_image = tf.image.resize(input_image, (224, 224))
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0

    label = model.predict(input_image)
    label = tf.argmax(label, axis=-1)
    label = tf.squeeze(label)
    label = label.numpy()

    return label