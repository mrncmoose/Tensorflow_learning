'''
Created on Nov 23, 2022

@author: moose
'''
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
import numpy as np
import os
import glob

batch_size = 32
img_height = 224
img_width = 224
imageDir = 'cameraImages'
canidateDir = imageDir + '/airplaneCanidates/'
threshold = 0.95

mobilenet_v2 = 'https://tfhub.dev/google/object_detection/mobile_object_labeler_v1/1' # need to figure out the input shape and what the error messages really mean.
# mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4' # not very consistant
module = hub.Module(mobilenet_v2)
IMAGE_SHAPE = (224, 224)
mobileNetShape = hub.get_expected_image_size(module)
classifier = tf.keras.Sequential([
    hub.KerasLayer(mobilenet_v2, input_shape=mobileNetShape)
])

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
imageFiles = glob.glob(imageDir + '/*.JPG')
for imageFile in imageFiles:
    try:
        img = Image.open(imageFile).resize(IMAGE_SHAPE)
        imgArray = np.array(img)/255.0
        img_expended_dim = imgArray[np.newaxis, ...]
        result = classifier.predict(img_expended_dim)
        predicted_class = tf.math.argmax(result[0], axis=-1)
        predicted_class_name = imagenet_labels[predicted_class]
        if predicted_class_name == 'Space Shuttle' or predicted_class_name == 'Airliner':
            print('Image {} predition is {}'.format(imageFile, predicted_class_name.title()))
        
        # if np.max(score) >= threshold and labels_path[np.argmax(score)] == 'airplanes':
        #     os.rename(imageDir + '/' + imageFile, canidateDir + imageFile)
        #     print("Image {} most likely belongs to {} with a {:.2f} percent confidence.".format(imageFile, class_names[np.argmax(score)], 100 * np.max(score)))
    except Exception as e:
        print('Unable to process file {} for reason {}'.format(imageFile, e))