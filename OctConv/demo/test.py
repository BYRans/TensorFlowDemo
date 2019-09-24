# mnist attention
import numpy as np

np.random.seed(1337)
from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, Dropout, Flatten, Dense
from keras.models import *
from malware_classification import common_process_data as read_data
from keras.layers import Input, LSTM, Bidirectional, Conv2D, Reshape
from OctConv.keras_octave_conv import OctaveConv2D
import keras

batch_size = 64
TIME_STEPS = 25
INPUT_DIM = 25
lstm_units = 128
num_classes = 15
epochs = 25
alpha = 0.25  # alpha of OctConv

# data pre-processing
# (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
(X_train, y_train), (X_test, y_test) = read_data.load_npz_data(
    "F:/数据集/Kim2016/malware_dataset/malware_dataset/attention_train_test_data.npz")
X_train = X_train.reshape(-1, 25, 25) / 255.  # why / 255?
X_test = X_test.reshape(-1, 25, 25) / 255.
# y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
print('X_train shape before trained LSTM model:', X_train.shape)
print('X_test shape before trained LSTM model:', X_test.shape)

trained_lstm_model = load_model("F:/数据集/Kim2016/malware_dataset/malware_dataset/mnist_attention_model.h5")

trained_lstm_model.summary()

trained_lstm_model_preout_model = Model(inputs=trained_lstm_model.input,
                                        outputs=trained_lstm_model.layers[8].output)

lstm_out_X_train_tmp = trained_lstm_model_preout_model.predict(X_train)
lstm_out_X_test_tmp = trained_lstm_model_preout_model.predict(X_test)

lstm_out_X_train = lstm_out_X_train_tmp.reshape(-1, 128, 100, 1)
lstm_out_X_test = lstm_out_X_test_tmp.reshape(-1, 128, 100, 1)

print("lstm_out_X_train_tmp shape after trained LSTM model:", lstm_out_X_train_tmp.shape)
print(lstm_out_X_train.shape)
print("y_train shape after trained LSTM model:", y_train.shape)

inputs = Input(shape=(128, 100, 1))

# build OctConv model linked attention LSTM outputs ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# Octave Conv
normal = BatchNormalization()(inputs)
high, low = OctaveConv2D(64, kernel_size=3)(normal)
high, low = MaxPool2D()(high), MaxPool2D()(low)
high, low = OctaveConv2D(32, kernel_size=3)([high, low])
conv = OctaveConv2D(16, kernel_size=3, ratio_out=0.0)([high, low])
pool = MaxPool2D()(conv)
flatten = Flatten()(pool)
normal = BatchNormalization()(flatten)
dropout = Dropout(rate=0.4)(normal)
outputs = Dense(units=15, activation='softmax')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
train_history = model.fit(
    x=lstm_out_X_train,
    y=y_train,
    epochs=epochs,
    validation_data=(lstm_out_X_test, y_test),
)
# octave_score = model.evaluate(lstm_out_X_test, y_test)
# print('Accuracy of Octave: %.4f' % octave_score[1])
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

print('Testing--------------')
loss, accuracy = model.evaluate(lstm_out_X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)

print("-----------------------DY Add------------------------")
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History of trained lstm-OctConv:')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(lstm_out_X_test, y_test)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))
