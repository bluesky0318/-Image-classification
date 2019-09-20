from model import get_model
from data import get_files
from config import config

def train():
    model = get_model()
    train_generator,validation_generator = get_files("train")

    model.fit_generator(
        train_generator,
        steps_per_epoch= config.nb_train_samples // config.batch_size,
        epochs= config.epochs,
        validation_data=validation_generator,
        validation_steps=config.nb_validation_samples // config.batch_size)

    model.save_weights('./checkpoints/first_try.h5')

def test():
    pass

if __name__ == '__main__':
    mode = "train"
    if mode == "train":
        train()
    elif mode == "test":
        test()
    else:
        print("check mode!")