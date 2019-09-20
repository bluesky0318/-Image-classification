### define global configs  ###

class DefaultConfigs(object):
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 400
    epochs = 50
    batch_size = 16
    img_width, img_height = 150, 150

config = DefaultConfigs()