'''
Created on Nov 23, 2022

@author: moose
'''
import tensorflow as tf
import numpy as np
import os 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataDir = 'data_sets/airplane_car_ship'
batch_size = 32
img_height = 224
img_width = 224

model = tf.keras.models.load_model('models/airplane_car_ship_model')

# imageFile = dataDir + '/test/airplanes/airplane11.jpg'
# imageFile = 'cardinal_back.jpg'
# imageFile = '592px-Red_sunflower.jpg'
# imageFile = 'data_sets/cats_and_dogs_images/cats_and_dogs_filtered/validation/cats/cat.2000.jpg'
imageFiles = [
    '592px-Red_sunflower.jpg',
    dataDir + '/train/airplanes/airplane11.jpg',
    dataDir + '/test/airplanes/airplane12.jpg',
    dataDir + '/test/airplanes/airplane25.jpg',
    'cardinal_back.jpg',
    dataDir + '/test/cars/cars3.jpg',
    dataDir + '/test/cars/cars14.jpg',
    dataDir + '/test/ships/2155186.jpg'
]

for imageFile in imageFiles:
    img = keras.preprocessing.image.load_img(imageFile, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['airplanes', 'cars', 'ship']
    print("Image {} most likely belongs to {} with a {:.2f} percent confidence.".format(imageFile, class_names[np.argmax(score)], 100 * np.max(score)))
