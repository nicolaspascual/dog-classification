import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path
import os
from keras.models import model_from_json

def plot_accuracy(history, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(base_folder, 'accuracy.pdf'))
    plt.close()


def plot_loss(history, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(base_folder, 'loss.pdf'))

def save_model(model, base_folder='./'):
    make_base_folder_if_needed(base_folder)
    model_json = model.to_json()
    with open(path.join(base_folder, 'model.json'), 'w') as json_file:
            json_file.write(model_json)
    weights_file = path.join(base_folder, 'weights.hdf5')
    model.save_weights(weights_file, overwrite=True)

def make_base_folder_if_needed(base_folder):
    try:
        os.makedirs(base_folder)
    except:
        pass