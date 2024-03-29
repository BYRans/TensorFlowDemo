# %%
from keras.utils import np_utils
from matplotlib import pyplot as plt
import time
from malware_classification.Self_Attention import Self_Attention_Layer
from malware_classification import common_process_data as read_data
from malware_classification import global_var as GLVAR
from keras.models import Model
from keras.layers import *


max_features = GLVAR.TOTAL_OPERATIONS_COUNT + 1 # 该数要比operation的个数大1
epochs=30
batch_size = 32

print('Loading data...')

(x_train, y_train), (x_test, y_test) = read_data.load_npz_data(GLVAR.TRAIN_AND_TEST_DATA)
X_train = x_train.reshape(-1,GLVAR.pic_pow_size * GLVAR.pic_pow_size)  # why / 255?
X_test = x_test.reshape(-1,GLVAR.pic_pow_size * GLVAR.pic_pow_size)
# 标签转换为独热码
y_train = np_utils.to_categorical(y_train, num_classes=GLVAR.NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, num_classes=GLVAR.NUM_CLASSES)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# %%数据归一化处理

maxlen = GLVAR.pic_pow_size * GLVAR.pic_pow_size

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)




S_inputs = Input(shape=(maxlen,), dtype='int32')

embeddings = Embedding(max_features, 256)(S_inputs)

O_seq = Self_Attention_Layer(256)(embeddings)

O_seq = GlobalAveragePooling1D()(O_seq)

O_seq = Dropout(0.5)(O_seq)

outputs = Dense(GLVAR.NUM_CLASSES, activation='softmax')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)

print(model.summary())
# try using different optimizers and different optimizer configs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
print('Train...')

h = model.fit(x_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test))

plt.plot(h.history["loss"], label="train_loss")
plt.plot(h.history["val_loss"], label="val_loss")
plt.plot(h.history["acc"], label="train_acc")
plt.plot(h.history["val_acc"], label="val_acc")
plt.legend()
plt.show()

print('Testing--------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)


print("-----------------------DY Add------------------------")
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    title = 'Train History of Self-Attention: epochs-' + str(epochs) + " " + str(time.strftime("%Y-%m-%d %X", time.localtime()))
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(h, 'acc', 'val_acc')

show_train_history(h, 'loss', 'val_loss')

scores = model.evaluate(X_test, y_test)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))


model.save("F:/数据集/ocatak/mal-api-2019/self_attention_model_256.h5")


