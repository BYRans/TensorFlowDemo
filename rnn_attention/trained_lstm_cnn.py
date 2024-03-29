# mnist attention
import numpy as np
np.random.seed(1337)
import time
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from malware_classification import common_process_data as read_data
from keras.layers import Input, LSTM, Bidirectional, Conv2D, Reshape

batch_size = 64
TIME_STEPS = 25
INPUT_DIM = 25
lstm_units = 128
num_classes = 15
epochs = 50

# data pre-processing
# (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
(X_train, y_train), (X_test, y_test) = read_data.load_npz_data("F:/数据集/Kim2016/malware_dataset/malware_dataset/attention_train_test_data.npz")
X_train = X_train.reshape(-1,25, 25) / 255. # why / 255?
X_test = X_test.reshape(-1,25, 25) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


trained_lstm_model = load_model("F:/数据集/Kim2016/malware_dataset/malware_dataset/mnist_attention_model.h5")

trained_lstm_model.summary()

trained_lstm_model_preout_model = Model(inputs=trained_lstm_model.input,
                                     outputs=trained_lstm_model.layers[8].output)


lstm_out_X_train_tmp = trained_lstm_model_preout_model.predict(X_train)
lstm_out_X_test_tmp = trained_lstm_model_preout_model.predict(X_test)

lstm_out_X_train = lstm_out_X_train_tmp.reshape(-1,128,100,1)
lstm_out_X_test = lstm_out_X_test_tmp.reshape(-1,128,100,1)

print(lstm_out_X_train_tmp.shape)
print(lstm_out_X_train.shape)
print(y_train.shape)

inputs = Input(shape=(128, 100, 1))
# build CNN model linked RNN outputs
x = Conv2D(filters=128,
           kernel_size=(5,5),
           padding='same',
           input_shape=(128,100, 1),
           activation='relu',
           name='conv2d_1')(inputs)
x = MaxPool2D(pool_size=(2,2), name='max_pooling2d_1')(x)
x = Conv2D(filters=64,
           kernel_size=(5,5),
           padding='same',
           input_shape=(128,100, 1),
           activation='relu',
           name='conv2d_2')(x)
x = MaxPool2D(pool_size=(2,2), name='max_pooling2d_2')(x)
x = Conv2D(filters=32,
           kernel_size=(5,5),
           padding='same',
           input_shape=(128,100, 1),
           activation='relu',
           name='conv2d_3')(x)
x = MaxPool2D(pool_size=(2,2), name='max_pooling2d_3')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(625)(x)
x = Dropout(0.5)(x)
# output = TimeDistributed(Dense(num_classes, activation='softmax'))(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=output)


# second way attention
# inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
# units = 32
# activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
#
# attention = Dense(1, activation='tanh')(activations)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(units)(attention)
# attention = Permute([2, 1], name='attention_vec')(attention)
# attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
# out_attention_mul = Flatten()(attention_mul)
# output = Dense(10, activation='sigmoid')(out_attention_mul)
# model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print('Training------------')
train_history = model.fit(lstm_out_X_train, y_train,validation_split=0.2, epochs=epochs, batch_size=batch_size)

print('Testing--------------')
loss, accuracy = model.evaluate(lstm_out_X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)


print("-----------------------DY Add------------------------")
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History of trained lstm-cnn:')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(lstm_out_X_test, y_test)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))


