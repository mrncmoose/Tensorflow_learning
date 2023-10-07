import unittest
from ProcessText import ProcessText

class TestProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.imageFile = '/Users/moose/Pictures/AugustPlaneImages/airPlaneCanidates/5MTC0045.JPG'
        self.pt = ProcessText()
        self.textFromImage = 'N6617H'
        return super().setUp()
    
    def TestProcess(self):
        tList = self.pt.Process()
        self.assertTrue(len(tList)>0)
        