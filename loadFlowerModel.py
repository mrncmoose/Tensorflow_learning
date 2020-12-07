'''
Created on Dec 6, 2020

@author: moose
'''
import tensorflow as tf
import numpy as np
import os 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataDir = 'data_sets/flower_photos'
batch_size = 32
img_height = 180
img_width = 180

model = tf.keras.models.load_model('models/flower_model')

sunflower_file = '592px-Red_sunflower.jpg'

img = keras.preprocessing.image.load_img(sunflower_file, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
