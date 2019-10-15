from waldo_CNN import *
from tensorflow.keras.utils.vis_utils import plot_model

waldo = WaldoCNN(50, 10, 'data/Keras Generated/Train',
                 'data/Keras Generated/Test', 1,
                 load_model=os_path.join(MODEL_DIRECTORY, 'model_v1.h5'))


class WaldoMetrics():
    ''' This class will plot metrics associated with a model defined'''

    def __init__(self, model):

        self.model = model
        self.summary = self.model.summary()
        self.score = self.model.score
        self.metrics = self.model.hist()