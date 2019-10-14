from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io, color, filters
import matplotlib.pyplot as plt

class WaldoGenerator(obj):
    ''' This class is for generating waldo pictures from a folder'''

    def __init__(self,import_path_name, export_path_name)
    self.import_path_name = import_path_name
    self.export_path_name = export_path_name
