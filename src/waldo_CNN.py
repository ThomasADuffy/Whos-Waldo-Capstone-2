import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model
from skimage import io, transform, color
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
import os
import sys
import csv

SCRIPT_DIRECTORY = os.path.realpath(__file__)
ROOT_DIRECTORY = os.path.split(os.path.split(SCRIPT_DIRECTORY)[0])[0]
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'model')
LOGDIR_DIRECTORY = os.path.join(MODEL_DIRECTORY, 'logdir')


class WaldoCNN():
    """This is the CNN class which will build the model and the dataset.

    Inputs:

        batchsize - the number of images to train befor updating weights
        train_data_loc - from the root directory, the path to the training data
        test_data_loc - from the root directory, the path to the testing data
        version - version of model
        load_model"""

    def __init__(self, batchsize, epochs, train_data_loc,
                 test_data_loc, holdout_data_loc, version, load_model=False):
        self.batchsize = batchsize
        self.epochs = epochs
        self.train_data_loc = os.path.join(ROOT_DIRECTORY, train_data_loc)
        self.test_data_loc = os.path.join(ROOT_DIRECTORY, test_data_loc)
        self.holdout_data_loc = os.path.join(ROOT_DIRECTORY, holdout_data_loc)
        self.version = version
        self.load_model = load_model
        self.create_dataset_generators()
        self.create_model()

    def create_dataset_generators(self):
        '''This will create the dataset generators for the model to use'''

        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.holdout_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
                self.train_data_loc,
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb',
                target_size=(64, 64),
                shuffle=True)
        self.validation_generator = self.test_datagen.flow_from_directory(
                self.test_data_loc,
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb',
                target_size=(64, 64))
        self.holdout_generator = self.holdout_datagen.flow_from_directory(
                self.holdout_data_loc,
                batch_size=self.batchsize,
                class_mode='binary',
                color_mode='rgb',
                target_size=(64, 64))

    def create_model(self):
        '''This will create the hard encoded model'''

        if self.load_model:
            self.model = load_model(os.path.join(MODEL_DIRECTORY,
                                                 self.load_model))
        else:
            self.model = Sequential()
            self.model.add(Conv2D(64, (2, 2), input_shape=(64, 64, 3),
                                  padding='valid',
                                  name='Convolution-1',
                                  activation='relu'))
            self.model.add(Conv2D(64, (2, 2), padding='valid',
                                  name='Convolution-2',
                                  activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2),
                                        name='Pooling-1'))
            self.model.add(Dropout(0.05))

            self.model.add(Conv2D(128, (2, 2), padding='valid',
                                  name='Convolution-3',
                                  activation='relu'))
            self.model.add(Conv2D(128, (2, 2), padding='valid',
                                  name='Convolution-4',
                                  activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2),
                                        name='Pooling-2'))
            self.model.add(Dropout(0.10))

            self.model.add(Flatten())
            self.model.add(Dense(256,
                                 name='Dense-1',
                                 activation='relu'))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(1,
                                 name='Dense-2',
                                 activation='sigmoid'))

            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy', Precision(), Recall()])

    def fit(self):
        ''' This will fit the model with the data inputed'''

        tensorboard = callbacks.TensorBoard(
            log_dir=LOGDIR_DIRECTORY,
            histogram_freq=0, 
            write_graph=True,
            update_freq='epoch')

        savename = os.path.join(MODEL_DIRECTORY, f"model_v{self.version}_best.h5")


        mc = callbacks.ModelCheckpoint(
            savename,
            monitor='val_recall', 
            verbose=0, 
            save_best_only=True, 
            mode='max', 
            save_freq='epoch')
        
        self.callbacklst = [mc]

        self.hist = self.model.fit_generator(self.train_generator,
                                             steps_per_epoch=None,
                                             epochs=self.epochs, verbose=1,
                                             callbacks=None,
                                             validation_data=self.validation_generator,
                                             validation_steps=None,
                                             validation_freq=1,
                                             class_weight=None,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             shuffle=True, initial_epoch=0)

        self.score_model()
        self.metrics = self.hist.history
        self.save_question()

    def save_model(self):
        '''This will save the models and weights to the Model directory'''

        self.model.save(os.path.join(MODEL_DIRECTORY,
                                     f'model_v{self.version}.h5'))
        with open(os.path.join(MODEL_DIRECTORY,
                               f'model_v{self.version}_metrics.csv'),
                  'w') as f:
            w = csv.DictWriter(f, self.metrics.keys())
            w.writeheader()
            for i in range(0, len(list(self.metrics.values())[0])):
                w.writerow({list(self.metrics.keys())[0]: list(self.metrics.values())[0][i],
                           list(self.metrics.keys())[1]: list(self.metrics.values())[1][i],
                           list(self.metrics.keys())[2]: list(self.metrics.values())[2][i],
                           list(self.metrics.keys())[4]: list(self.metrics.values())[4][i],
                           list(self.metrics.keys())[5]: list(self.metrics.values())[5][i],
                           list(self.metrics.keys())[6]: list(self.metrics.values())[6][i],
                           list(self.metrics.keys())[7]: list(self.metrics.values())[7][i]})
        print("Saved model and metrics to disk")

    def score_model(self):
        '''This will score the model on a small hold out set
        and return the values'''

        self.score = self.model.evaluate(self.holdout_generator, verbose=0)
        print('Tested on holdout set:')
        print('Test score:', self.score[0])
        print('Test accuracy:', self.score[1])
        print('Test percision:', self.score[2])
        print('Test recall:', self.score[3])

    def save_question(self):
        ''' This will ask to save model and will save it prompted too'''

        anslstY = ['yes', 'y']
        anslstN = ['no', 'n']
        ans = input("Save model??? ")
        if type(ans) != str:
            print('Not valid ans, Please enter yes or no')
            self.save_question()
        else:
            if ans.lower() in anslstY:
                self.save_model()
            else:
                print('Model not saved.')


if __name__ == '__main__':
    waldo = WaldoCNN(15, 5, 'data/Keras Generated/New_Train',
                     'data/Keras Generated/Test',
                     'data/Keras Generated/Holdout', 3)
                   
    response = input('fit?')
    if response.lower() == 'y':
        waldo.fit()
