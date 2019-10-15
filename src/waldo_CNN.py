import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from skimage import io, transform, color
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import sys
np.random.seed(1337)

SCRIPT_DIRECTORY = os.path.realpath(__file__)
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'model')
sys.path.append(ROOT_DIRECTORY)


class WaldoCNN(object):
    """This is the CNN class which will build the model and the dataset.

    Inputs:

        batchsize - the number of images to train befor updating weights
        train_data_loc - from the root directory, the path to the training data
        test_data_loc - from the root directory, the path to the testing data
        version - version of model
        load_model"""

    def __init___(self, batchsize, epochs, train_data_loc,
                  test_data_loc, version, load_model=False):
        self.batchsize = batchsize
        self.epochs = epochs
        self.train_data_loc = os.path.join(ROOT_DIRECTORY, train_data_loc)
        self.test_data_loc = os.path.join(ROOT_DIRECTORY, test_data_loc)
        self.version = version
        self.load_model = load_model
        self.create_dataset_generators()
        self.create_model()

    def create_dataset_generators(self):
        '''This will create the dataset generators for the model to use'''

        self.train_data = ImageDataGenerator(rescale=1./255)
        self.test_data = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
                self.train_data_loc,
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb')
        self.validation_generator = self.test_datagen.flow_from_directory(
                self.test_data_loc,
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb')

    def create_model(self):
        '''This will create the hard encoded model'''
        if self.load_model:
            self.model = load_model(os.path.join(MODEL_DIRECTORY,
                                                 self.load_model))
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),
                           padding='valid'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(32, (3, 3), padding='valid'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(64, (3, 3), padding='valid'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(64))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))

            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])

    def fit(self):
        ''' This will fit the model with the data inputed'''

        self.model.fit_generator(self.train_generator, steps_per_epoch=None,
                                 epochs=self.epochs, verbose=1, callbacks=None,
                                 validation_data=self.validation_generator,
                                 validation_steps=None, validation_freq=1,
                                 class_weight=None, max_queue_size=10,
                                 workers=1, use_multiprocessing=False,
                                 shuffle=True, initial_epoch=0)
        self.score_model()

    def save_model(self):
        '''This will save the models and weights'''

        self.model.save(os.path.join(MODEL_DIRECTORY,
                                     f'model_v{self.version}.h5'))
        print("Saved model to disk")

    def score_model(self):
        '''This will score the model and return the values'''
        self.score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', self.score[0])
        print('Test accuracy:', self.score[1])


if __name__ == '__main__':
    waldo1 = WaldoCNN(50, 10, 'data/Keras Generated/Train',
                      'data/Keras Generated/Test', 1)
    waldo1.fit()

