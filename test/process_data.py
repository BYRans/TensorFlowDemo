import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from keras.utils import np_utils


def create_oprations_set(raw_filename, op_index_filename):
    with open(raw_filename) as raw_data:
        operations = set()
        for line in raw_data:
            operation = line.replace('\n', '').replace('"', '').split(',')
            del operation[0]
            del operation[0]
            operations = operations.union(set(operation))

    operations_list = list(operations)
    map_index = range(len(operations))
    ziped_op_index = zip(operations_list, map_index)

    operations_dic = {k: v for k, v in ziped_op_index}

    with open(op_index_filename, 'w') as json_file:
        json.dump(operations_dic, json_file, ensure_ascii=False)
    print("operations index dictionary create success! Dic file saved in ", op_index_filename)
    print("the operations's count is:",len(operations))


def raw_labels_to_one_hot(raw_lables_list):
    for i, lable in enumerate(raw_lables_list):
        raw_lables_list[i] = lable.split(".")[0]

    lables_set_index = {}
    for lable in raw_lables_list:
        if lable not in lables_set_index:
            lables_set_index[lable] = len(lables_set_index)
    print("the lables's count is:",len(lables_set_index))
    lables_index_np = np.zeros(len(raw_lables_list))
    for i, lable in enumerate(raw_lables_list):
        lables_index_np[i] = lables_set_index.get(lable)
    lables_one_hot = np_utils.to_categorical(lables_index_np)
    return lables_one_hot


def process_raw_data(raw_filename, op_index_filename):

    with open(op_index_filename, 'r') as fileR:
        operation_dic = json.load(fileR)
        fileR.close()

    with open(raw_filename) as raw_data:
        line_num = len(raw_data.readlines())

    with open(raw_filename) as raw_data:
        length = pow(math.ceil(math.sqrt(len(operation_dic))), 2)
        print("the total operations's ocunt is:",len(operation_dic),"the picture size is",math.ceil(math.sqrt(len(operation_dic))),"*",math.ceil(math.sqrt(len(operation_dic))))
        processed_data_np = np.empty(shape=(line_num, length)).astype("int32")
        labels_list = []
        for i, line in enumerate(raw_data):
            tmp_processed_data = [0 for x in range(0, length)]
            operation = line.replace('\n', '').replace('"', '').split(',')
            labels_list.append(operation.pop(0))
            del operation[0]
            operation_set = set(operation)
            for op in operation_set:
                if len(op) != 0:
                    index = operation_dic[op]
                    tmp_processed_data[index] = 1
            processed_data_np[i] = np.array(tmp_processed_data)

        labels_index_np = raw_labels_to_index(labels_list)

        x_train, x_test, y_train, y_test = train_test_split(processed_data_np, labels_index_np, test_size=0.01,
                                                            random_state=0)

        np.savez('F:/数据集/Kim2016/malware_dataset/malware_dataset/train_test_data.npz', x_train=x_train, x_test=x_test,
                 y_train=y_train, y_test=y_test)


def load_npz_data(file_path):
    f = np.load(file_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def cnn_train(data_file_path):
    (X_Train, y_Train), (X_Test, y_Test) = load_npz_data(data_file_path)

    # Translation of data
    X_Train4D = X_Train.reshape(X_Train.shape[0], 33, 33, 1).astype('float32')
    X_Test4D = X_Test.reshape(X_Test.shape[0], 33, 33, 1).astype('float32')

    # Standardize feature data
    X_Train4D_norm = X_Train4D / 255
    X_Test4D_norm = X_Test4D / 255

    # Label Onehot-encoding
    y_TrainOneHot = np_utils.to_categorical(y_Train)
    y_TestOneHot = np_utils.to_categorical(y_Test)

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

    model = Sequential()
    # Create CN layer 1
    model.add(Conv2D(filters=256,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=(33, 33, 1),
                     activation='relu',
                     name='conv2d_1'))
    # Create Max-Pool 1
    model.add(MaxPool2D(pool_size=(2, 2), name='max_pooling2d_1'))

    # Create CN layer 2
    model.add(Conv2D(filters=128,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=(33, 33, 1),
                     activation='relu',
                     name='conv2d_2'))

    # Create Max-Pool 2
    model.add(MaxPool2D(pool_size=(2, 2), name='max_pooling2d_2'))

    # Create CN layer 3
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=(33, 33, 1),
                     activation='relu',
                     name='conv2d_3'))

    # Create Max-Pool 3
    model.add(MaxPool2D(pool_size=(2, 2), name='max_pooling2d_3'))



    # Add Dropout layer
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Flatten(name='flatten_1'))

    model.add(Dense(128, activation='relu', name='dense_1'))
    model.add(Dropout(0.5, name='dropout_2'))

    model.add(Dense(15, activation='softmax', name='dense_2'))

    model.summary()
    print("")

    # 定義訓練方式
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 開始訓練
    train_history = model.fit(x=X_Train4D_norm,
                              y=y_TrainOneHot, validation_split=0.2,
                              epochs=100, batch_size=300, verbose=1)

    import matplotlib.pyplot as plt


    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    show_train_history(train_history, 'acc', 'val_acc')

    show_train_history(train_history, 'loss', 'val_loss')

    scores = model.evaluate(X_Test4D_norm, y_TestOneHot)
    print()
    print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))

    print("\t[Info] Making prediction of X_Test4D_norm")
    prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction
    print()
    print("\t[Info] Show 10 prediction result (From 240):")
    print("%s\n" % (prediction[240:250]))






class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(
      train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  validation = DataSet(
      validation_images,
      validation_labels,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  test = DataSet(
      test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)





def main():
    raw_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/malware_API_dataset - bak - selected.csv'
    op_index_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/operations_map.json'

    create_oprations_set(raw_filename, op_index_filename)
    process_raw_data(raw_filename, op_index_filename)



if __name__ == "__main__":
    main()
