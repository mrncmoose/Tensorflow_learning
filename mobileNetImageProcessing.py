from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
import numpy as np
import imageio.v2 as imageio
import PIL.Image as Image
import glob
from pathlib import Path

#TODO:  What does mobile net think image 5MTC0147.JPG is?'

imageDir = 'cameraImages'
canidateDir = imageDir + '/airplaneCanidates/'
canidateLabels = ['white_stork', 'warplane', 'space_shuttle', 'wing', 'airliner']
img_height = 224
img_width = 224
model = MobileNetV2(weights='imagenet')
imageFiles = glob.glob(imageDir + '/*.JPG')
for imageFile in imageFiles:
    try:
        data = np.empty((1, 224, 224, 3))

        img = imageio.imread(imageFile)
        data[0] = Image.fromarray(img).resize((img_height, img_width))
        data = preprocess_input(data)
        predictions = model.predict(data)
        output_neuron = np.argmax(predictions[0])
        for name, desc, score in decode_predictions(predictions)[0]:
            if desc in canidateLabels and score > 0.1:
                Path(imageFile).rename(canidateDir + imageFile)
                print('Image - {} has a prediction of {} with {:.2f}%% accuracy'.format(imageFile, desc, 100 * score))
                break
        
    except Exception as e:
        print('Unable to process file {} for reason {}'.format(imageFile, e))
