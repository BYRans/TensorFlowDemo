import numpy as np
from keras.utils import np_utils

np.random.seed(1337)
from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, Dropout, Flatten, Dense
from keras.models import *
from malware_classification import common_process_data as read_data
from keras.layers import Input, LSTM, Bidirectional, Conv2D, Reshape
from OctConv.keras_octave_conv import OctaveConv2D, octave_dual
import keras


def load_mal60npz_data(file_path):
    f = np.load(file_path)
    data, lables = f['data.npy'], f['labels.npy']
    f.close()
    return (data, lables)


(data, lables) = load_mal60npz_data("F:/数据集/mal60/mal60.npz")

print(data)
