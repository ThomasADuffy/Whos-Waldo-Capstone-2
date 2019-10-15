from os import listdir, remove
import os
import sys
import unittest
from tensorflow.keras.preprocessing.image import ImageDataGenerator
SCRIPT_DIRECTORY = os.path.realpath(__file__)
ROOT_DIRECTORY = os.path.split(os.path.split(SCRIPT_DIRECTORY)[0])[0]
SRC_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
TEST_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'test')
sys.path.append(SRC_DIRECTORY)
sys.path.append(ROOT_DIRECTORY)
import waldo_generator


class TestWaldoGenorator(unittest.TestCase):

    def setUp(self):
        self.Waldo = waldo_generator.WaldoGenerator(os.path.join(TEST_DIRECTORY,
                                                                'img_test'),
                                                    os.path.join(TEST_DIRECTORY,
                                                                'output'), 1)

    def test_imgs_count(self):
        '''Check to see if images are the same produced.'''
        self.assertEqual(len(listdir(os.path.join(TEST_DIRECTORY, 'output'))),
                         self.Waldo.num_images)
        for x in listdir(os.path.join(TEST_DIRECTORY, 'output')):
            remove(os.path.join(os.path.join(TEST_DIRECTORY, 'output'), x))
            print(f"{x} Removed!")


if __name__ == '__main__':
    unittest.main()
