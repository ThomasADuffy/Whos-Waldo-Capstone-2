import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from skimage import io, color, filters
from tensorflow.keras.utils import plot_model
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
        '''This will plot the accuracy and loss plots for the model'''

        fig, ax = plt.subplots(1, 2, figsize=(12, 9))
        ax[0].plot(self.df['accuracy'], lw=3, marker='.')
        ax[0].plot(self.df['val_accuracy'], lw=3, marker='.')
        ax[0].set_title('Model accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Test'], loc='upper left')

        ax[1].plot(self.df['loss'], lw=3, marker='.')
        ax[1].plot(self.df['val_loss'], lw=3, marker='.')
        ax[1].set_title('Model loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Test'], loc='upper left')
        a = self.score_accuracy
        ls = self.score_loss
        fig.suptitle(f'Model V{self.waldo.version} Loss:{ls} Acc:{a} (on holdout)',
                     fontsize=18)
        plt.savefig(figloc)

    def plot_model_structure(self, savedir):
        ''' This will plot a model structure using keras's built in viz func'''

        return plot_model(self.waldo.model, to_file=savedir,
                          show_shapes=True, expand_nested=True)

    def save_wrong_imgs(self, savedir):
        hold_x, hold_y = self.waldo.holdout_generator[0]
        valid_x, valid_y = self.waldo.holdout_generator[0]
        predict_array_hold = (np.where(self.waldo.model.predict(hold_x) >= .5, 1, 0)).reshape(1, -1)
        predict_array_valid = (np.where(self.waldo.model.predict(valid_x) >= .5, 1, 0)).reshape(1, -1)
        y_true_hold = hold_y.reshape(1, -1)
        y_true_valid = valid_y.reshape(1, -1)
        idx_lst_hold = list(np.where(predict_array_hold != y_true_hold)[1])
        idx_lst_valid = list(np.where(predict_array_valid != y_true_valid)[1])
        for i, idx in enumerate(idx_lst_hold):
            io.imsave(fname=os.join.path(savedir, f'Model_V{self.waldo.version}_hold{i+1}.jpg', arr=hold_x[idx]))
        for i, idx in enumerate(idx_lst_valid):
            io.imsave(fname=os.join.path(savedir, f'Model_V{self.waldo.version}_valid{i+1}.jpg', arr=hold_x[idx]))
        self.wronginfo_hold = {'idxlst': idx_lst_hold,
                               'problst': self.waldo.model.predict(hold_x)[idx_lst_hold],
                               'imglst': hold_x[idx_lst_hold]}
        self.wronginfo_valid = {'idxlst': idx_lst_valid,
                                'problst': self.waldo.model.predict(valid_x)[idx_lst_valid],
                                'imglst': hold_x[idx_lst_valid]}


if __name__ == '__main__':
    pass
