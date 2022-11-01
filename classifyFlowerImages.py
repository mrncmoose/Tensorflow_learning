#  Now that the model is trained & works, see how it works on images outside of the exercise.
'''
Created on Oct 19, 2022

@author: moose
'''
import tensorflow as tf
import numpy as np
import os 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

if __name__ == '__main__':
    dataDir = '/Users/moose/Pictures/flowerPixs'
    batch_size = 32
    img_height = 180
    img_width = 180

    model = tf.keras.models.load_model('models/flower_model')
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    for root, dirs, files in os.walk(dataDir):
        for f in files:
            if f.endswith(('.jpg','.jpeg', '.JPG', '.JPEG')): 
                print('Working with file: {}'.format(f))
                imageFile = dataDir + '/' + f
                try:      
                    img = keras.preprocessing.image.load_img(imageFile, target_size=(img_height, img_width))
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) # Create a batch
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
                except Exception as e:
                    print('Unable to process file {} for reason {}'.format(f, e))
                    