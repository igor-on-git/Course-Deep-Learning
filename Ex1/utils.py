import pathlib
import os
import shutil

import numpy as np

from matplotlib import image
import matplotlib.pyplot as plt

def load_data(data_folder, split):

    desktop = pathlib.Path(data_folder)
    data_filenames = [str(item) for item in desktop.rglob('*.png') if item.is_file()]

    np.random.seed(1)
    rand_order = np.random.permutation(len(data_filenames))
    data_filenames_rand = [data_filenames[i] for i in rand_order]

    train_len = int(split[0]*len(data_filenames_rand))
    train_files, rest_files = data_filenames_rand[:train_len], data_filenames_rand[train_len:]

    valid_len = int(split[1]*len(data_filenames_rand))
    valid_files, test_files = rest_files[:valid_len], rest_files[valid_len:]

    return train_files, valid_files, test_files


def reorder_data_for_image_folder(source, dest):

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest + '/0/', exist_ok=True)
    os.makedirs(dest + '/1/', exist_ok=True)
    for ii, fname in enumerate(source):
        if image.imread(fname).shape == (50, 50, 3):
            if fname[-5] == '1':
                shutil.copy2(fname, dest + '/1/')
            else:
                shutil.copy2(fname, dest + '/0/')


def plot_train_results(model_name, train_perf):

    plt.figure(1)
    plt.plot(range(len(train_perf['train_loss'])), train_perf['train_loss'], label='Train Loss')
    plt.plot(range(len(train_perf['valid_loss'])), train_perf['valid_loss'], label='Validation Loss')
    plt.title(model_name + ' Train and Validation loss')
    plt.legend()
    plt.grid(which='both', axis='both')
    plt.savefig('models/' + model_name + '/train_valid_loss.png')
    plt.close(1)

    plt.figure(2)
    plt.plot(range(len(train_perf['accuracy'])), train_perf['accuracy'], label='Accuracy')
    plt.plot(range(len(train_perf['precision'])), train_perf['precision'], label='Precision')
    plt.plot(range(len(train_perf['recall'])), train_perf['recall'], label='Recall')
    plt.plot(range(len(train_perf['F1score'])), train_perf['F1score'], label='F1 score')
    plt.title(model_name + ' Training metrics')
    plt.legend()
    plt.grid(which='both', axis='both')
    plt.savefig('models/' + model_name + '/train_perf.png')
    plt.close(2)
