import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from skimage import io, transform, color
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import sys
np.random.seed(1337)

SCRIPT_DIRECTORY = os.path.realpath(__file__)
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
sys.path.append(ROOT_DIRECTORY)
sys.path.append(DATA_DIRECTORY)


class WaldoCNN(object):
    "This is the physical CNN class"
    def __init___(self, batchsize):
        self.batchsize = batchsize

    def create_dataset(self):
        self.train_data = ImageDataGenerator(rescale=1./255)
        self.test_data = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
                'data/Keras Generated/Train',
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb')
        self.validation_generator = self.test_datagen.flow_from_directory(
                'data/Keras Generated/Test',
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb')

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3),padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3),padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    def fit(self):
        self.model.fit_generate
    fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)



model.save('model_v1.h5')
model.save_weights('weights_v1.h5')  

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1]) 