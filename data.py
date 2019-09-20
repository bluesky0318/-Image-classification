import os
from keras.preprocessing.image import ImageDataGenerator
from config import config

def get_files(mode):
    # for test
    if mode == "test":
        pass
    elif mode == "train": #Load all pictures
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            config.train_data_dir,
            target_size=(config.img_width, config.img_height),
            batch_size=config.batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            config.validation_data_dir,
            target_size=(config.img_width, config.img_height),
            batch_size=config.batch_size,
            class_mode='binary')
        return train_generator,validation_generator
    else:
        print("check the mode please!")