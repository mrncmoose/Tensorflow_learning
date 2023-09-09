from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
import numpy as np
import imageio.v2 as imageio
import PIL.Image as Image
import glob
from pathlib import Path
import os
import argparse

#TODO: Add ability to attempt to 'read' tail number or licence plate.

parser = argparse.ArgumentParser(description='AI image classifier for finding airplane images and cars.')
parser.add_argument('--imageDir', 
                    required=True,
                    default='cameraImages',
                    help='The directory with the images to process')
parser.add_argument('--baseDir',
                    default='.',
                    help='The base directory where filtered images will be placed.')
parser.add_argument('--airplaneCanidateDir',
                    default='airplaneImages',
                    help='The directory from the base directory where airplane images will be placed.')
parser.add_argument('--carDir',
                    default='carCanidateImages',
                    help='The directory from the base directory where car images will be placed')

args = parser.parse_args()

imageDir = args.imageDir
canidateDir = args.baseDir + '/' + imageDir + '/' + args.airplaneCanidateDir + '/'
carCanidateDir = args.baseDir + '/' + imageDir + '/' + args.carDir + '/'

if not Path.exists(imageDir):
    print('Base image dir {} does not exisit.  Exiting'.format(imageDir))
    exit -1
    
if not Path.exists(args.baseDir):
    print('Base dir of {} does not exisit. Creating...'.format(args.baseDir))
    try:
        Path.mkdir(args.baseDir)
    except Exception as e:
        print('Unable to create base dir of {}'.format(args.baseDir))
        exit -1
if not Path.exists(canidateDir):
    print('Airplane canidate dir {} missing.  Creating...'.format(canidateDir))
    Path.mkdir(canidateDir)

if not Path.exists(carCanidateDir):
    print('Car canidate dir {} missing.  Creating...'.format(carCanidateDir))
    Path.mkdir(carCanidateDir)
    
planeImageCount = len(os.listdir(canidateDir))
carImageCount = len(os.listdir(carCanidateDir))
if (planeImageCount != 0):
    print('Warning: Canidate plane images exisit in dir {}!'.format(canidateDir))

if (carImageCount != 0):
    print('Warning:  Car canidate images exisit in directory {}!'.format(carCanidateDir))

canidateLabels = ['white_stork', 'warplane', 'space_shuttle', 'wing', 'airliner']
carLabels = ['minivan', 'jeep']
img_height = 224
img_width = 224
model = MobileNetV2(weights='imagenet')
imageFiles = glob.glob(imageDir + '/*.JPG')
planeImgCount = 0
carImgCount = 0
for imageFile in imageFiles:
    try:
        data = np.empty((1, 224, 224, 3))

        img = imageio.imread(imageFile)
        data[0] = Image.fromarray(img).resize((img_height, img_width))
        data = preprocess_input(data)
        predictions = model.predict(data)
        output_neuron = np.argmax(predictions[0])
        for name, desc, score in decode_predictions(predictions)[0]:
            if desc in canidateLabels and score > 0.044:
                planeImgCount += 1
                Path(imageFile).rename(canidateDir + imageFile)
                print('Image - {} has a prediction of {} with {:.2f}%% accuracy'.format(imageFile, desc, 100 * score))
                break
            if desc in carLabels and score > 0.01:
                carImgCount += 1
                Path(imageFile).rename(carCanidateDir + imageFile)
                print('Image - {} has a prediction of {} with {:.2f}%% accuracy'.format(imageFile, desc, 100 * score))
                break
        
    except Exception as e:
        print('Unable to process file {} for reason {}'.format(imageFile, e))

print('Found {} airplane and {} car images'.format(planeImgCount, carImgCount))
