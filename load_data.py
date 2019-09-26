from keras.preprocessing.image import ImageDataGenerator
from os import path

def load_data(base_folder, img_rows, img_cols):

    train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=path.join(base_folder, 'train/'),
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    valid_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=path.join(base_folder, 'validation/'),
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=path.join(base_folder, 'test/'),
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, valid_generator, test_generator
