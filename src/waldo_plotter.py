import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
        ls = self.score_los
        fig.suptitle(f'Model V{self.waldo.version} Loss:{ls} Acc:{a} (on holdout)',
                     fontsize=18)
        plt.savefig(figloc)

    def plot_model_structure(self, savedir):
        return plot_model(self.waldo.model, to_file=savedir,
                          show_shapes=True, expand_nested=True)

    def plot_wrong_imgs(self, savedir):
        pass


if __name__ == '__main__':
    pass