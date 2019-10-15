from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
SCRIPT_DIRECTORY = os.path.realpath(__file__)
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
sys.path.append(ROOT_DIRECTORY)


class WaldoGenerator(object):
    ''' This class is for generating waldo pictures from a folder'''

    def __init__(self, import_path_name, export_path_name, numimages):
        self.import_path_name = import_path_name
        self.export_path_name = export_path_name
        self.num_images = numimages
        self.create_and_save_waldo_imgs()

    def create_img_generator(self):
        ''' this creates the image generator object'''
        self.datagen = ImageDataGenerator(
                                            rotation_range=0,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            rescale=1./255,
                                            shear_range=0,
                                            zoom_range=0.2,
                                            brightness_range=(.7, 1.3),
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            fill_mode='nearest')

    def create_pictures(self):
        '''This will run through the generator created
        in the create_img_generator function to actually save the images'''

        print(f'Loading pictures from {self.import_path_name}')
        i = 0
        for batch in self.datagen.flow_from_directory(
                                        directory=self.import_path_name,
                                        save_to_dir=self.export_path_name,
                                        save_prefix='keras_',
                                        save_format='jpg',
                                        target_size=(64, 64),
                                        batch_size=1, color_mode='rgb',
                                        interpolation='nearest'):
            i += 1
            if i == self.num_images:
                break
        print(f'''Saved and augmented {i} batches of pictures in
         {self.export_path_name}''')

    def create_and_save_waldo_imgs(self):
        '''This runs both functions to do all of it in one method'''
        self.create_img_generator()
        self.create_pictures()


if __name__ == '__main__':

    WaldoGenerator(os.path.join(ROOT_DIRECTORY, 'data/Keras Generated/Train'),
                   os.path.join(ROOT_DIRECTORY,
                                'data/Keras Generated/Train/waldo/generated'),
                   3950)

    WaldoGenerator(os.path.join(ROOT_DIRECTORY, 'data/Keras Generated/Test'),
                   os.path.join(ROOT_DIRECTORY,
                                'data/Keras Generated/Test/waldo/generated'),
                   987)
