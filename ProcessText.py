import cv2
import pytesseract
import argparse

class ProcessText:
    def __init__(self, imageFile):
        self.imageFile = imageFile
        self.image = cv2.imread(imageFile)
        self.tesseractBin = '/opt/homebrew/Cellar/tesseract/5.3.2_1'
        self.tesseractShape = 10
        self.kernelSize = 10
        
    def Process(self):
        greyScaleImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(greyScaleImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.tesseractShape, self.kernelSize))
        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
 
        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)
 
        # Creating a copy of image
        im2 = self.image.copy()   #FTD:  The image will croped & modified below.  Need to feed the full images back into the process.
 
 
        # Looping through the identified contours
        # Then rectangular part is cropped and passed on
        # to pytesseract for extracting text from it
        foundTextList = list()
        text:str = ''
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Drawing a rectangle on copied image
            rect = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
            foundText = pytesseract.image_to_string(cropped)
            # text = text + pytesseract.image_to_string(cropped)
            if len(foundText)>0:
                foundTextList.append(foundText)
                print('Found text: {}'.format(foundText))
        return foundTextList
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI process image for text.')
    parser.add_argument('--imageName', 
                    required=True,
                    default='cameraImages',
                    help='The image file with text to process')
    args = parser.parse_args()
    p = ProcessText(args.imageName)
    textList = p.Process()
    if len(textList) > 0:
        for t in textList:
            print('Found text: {}'.format(t))
            