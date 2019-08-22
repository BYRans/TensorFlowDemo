from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

np.random.seed(10)
l=[0 for x in range(0, 10)]
a=[0 for x in range(0, len(l))]
print(a)

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([0, 1])
np.savez('F:/数据集/mnist/test.npz', X_train=x, Y_train=y)
f = np.load('F:/数据集/mnist/test.npz')
x_train, y_train = f['X_train'], f['Y_train']
f.close()
print(x_train)
print(y_train)




Read MNIST data
path = 'F:/数据集/mnist/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
print(type(f))

# Translation of data
X_Train4D = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')
X_Test4D = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')




# Standardize feature data
X_Train4D_norm = X_Train4D / 255
X_Test4D_norm = X_Test4D / 255

# Label Onehot-encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)



np.set_printoptions(threshold=np.inf)
print(X_Test4D_norm[0])
