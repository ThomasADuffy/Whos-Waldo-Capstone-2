import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from skimage import io, color, filters
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from waldo_CNN import *

plt.style.use('ggplot')

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)


class WaldoPlotter():
    '''This is a plotting function
    Inputs:
    df - the dataframe of metrics from the csv
    score_loss - loss metric created by the holdout set
    score_accuracy - accuracy metric created by holdout set
    verson - verson of the model'''

    def __init__(self, df, waldo, score):
        self.df = df
        self.score_loss = score[0]
        self.score_accuracy = score[1]
        self.waldo = waldo

    def create_accuracy_loss(self, figloc):
        '''This will plot the accuracy and loss plots for the model
        Input:
        figloc- this is where it will save the plot'''

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].plot(self.df['accuracy'], lw=3, marker='.')
        ax[0].plot(self.df['val_accuracy'], lw=3, marker='.')
        ax[0].set_title('Model accuracy')
        ax[0].set_ylabel('Accuracy', fontsize=18)
        ax[0].set_xlabel('Epoch', fontsize=18)
        ax[0].legend(['Train', 'Test'], loc='upper left')

        ax[1].plot(self.df['loss'], lw=3, marker='.')
        ax[1].plot(self.df['val_loss'], lw=3, marker='.')
        ax[1].set_title('Model loss')
        ax[1].set_ylabel('Loss', fontsize=18)
        ax[1].set_xlabel('Epoch', fontsize=18)
        ax[1].legend(['Train', 'Test'], loc='upper left')
        a = self.score_accuracy
        ls = self.score_loss
        fig.suptitle(f'Model V{self.waldo.version} Loss:{ls} Acc:{a} (on holdout)',
                     fontsize=18)
        plt.savefig(figloc)

    def plot_model_structure(self, savedir):
        ''' This will plot a model structure using keras's built in viz func
        Input:
        savedir - this is where it will save the plot'''

        return plot_model(self.waldo.model, to_file=savedir,
                          show_shapes=True, expand_nested=True)

    def find_wrong_imgs(self, savedir):
        ''' This Function saves all of the images that it predicted wrong
        in the hold out set and the validation with titles of the
        probabilities, predicit, and actual.

        Input:
        Savedir - this is where the images w ill be saved

        both of the new attributes created also are just dictionaries that
        hold the info of the idx's where it got it wrong,
         images, and probabillities'''

        hold_x, hold_y = self.waldo.holdout_generator[0]
        valid_x, valid_y = self.waldo.validation_generator[0]
        predict_array_hold = (np.where(self.waldo.model.predict(hold_x) >= .5, 1, 0)).reshape(1, -1)
        predict_array_valid = (np.where(self.waldo.model.predict(valid_x) >= .5, 1, 0)).reshape(1, -1)
        y_true_hold = hold_y.reshape(1, -1)
        y_true_valid = valid_y.reshape(1, -1)
        idx_lst_hold = list(np.where(predict_array_hold != y_true_hold)[1])
        idx_lst_valid = list(np.where(predict_array_valid != y_true_valid)[1])
        self.wronginfo_hold = {'idxlst': idx_lst_hold,
                               'problst': self.waldo.model.predict(hold_x)[idx_lst_hold],
                               'imglst': hold_x[idx_lst_hold]}
        self.wronginfo_valid = {'idxlst': idx_lst_valid,
                                'problst': self.waldo.model.predict(valid_x)[idx_lst_valid],
                                'imglst': valid_x[idx_lst_valid]}

        for i, idx in enumerate(idx_lst_hold):
            y_p = predict_array_hold[:, idx]
            y_t = y_true_hold[:, idx]
            metric = self.metric_finder(y_p[0], y_t[0])
            prob = self.wronginfo_hold['problst'][i]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(X=hold_x[idx])
            ax.set_title(f"Probabillity={prob} Predict={y_p} Actual={y_t} | {metric}")
            plt.savefig(os.path.join(savedir, f'Model_V{self.waldo.version}_hold{i}.jpg'))
            plt.close(fig)

        for i, idx in enumerate(idx_lst_valid):
            y_p = predict_array_valid[:, idx]
            y_t = y_true_valid[:, idx]
            metric = self.metric_finder(y_p[0], y_t[0])
            prob = self.wronginfo_valid['problst'][i]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(X=valid_x[idx])
            ax.set_title(f"Probabillity={prob} Predict={y_p} Actual={y_t} | {metric}")
            plt.savefig(os.path.join(savedir, f'Model_V{self.waldo.version}_valid{i}.jpg'))
            plt.close(fig)

    def metric_finder(self, predict, actual):
        '''Helper function to find if an image is
        a False negative or False postivie'''

        if predict > actual:
            return 'FP'
        else:
            return 'FN'



if __name__ == '__main__':
    pass
