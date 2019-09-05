import numpy as np
import json
from sklearn.model_selection import train_test_split


def raw_labels_to_index(raw_lables_list):
    for i, lable in enumerate(raw_lables_list):
        raw_lables_list[i] = lable.split(".")[0]

    lables_set_index = {}
    for lable in raw_lables_list:
        if lable not in lables_set_index:
            lables_set_index[lable] = len(lables_set_index)
    print("the lables's count is:", len(lables_set_index))
    lables_index_np = np.zeros(len(raw_lables_list))
    for i, lable in enumerate(raw_lables_list):
        lables_index_np[i] = lables_set_index.get(lable)
    return lables_index_np


def process_raw_data(raw_filename, op_index_filename):
    np.set_printoptions(threshold=np.inf)

    with open(op_index_filename, 'r') as fileR:
        operation_dic = json.load(fileR)
        fileR.close()

    with open(raw_filename) as raw_data:
        line_num = len(raw_data.readlines())
        print("line_num:", line_num)

    with open(raw_filename) as raw_data:
        longest_operation_size = 0
        for line in enumerate(raw_data):
            operation = str(line).replace('\n', '').replace('(', '').replace(')', '').replace('"', '').split(',')
            tmp_len = len(operation) - 3
            if tmp_len > longest_operation_size:
                longest_operation_size = tmp_len
        print("longest_length is:", longest_operation_size)

    with open(raw_filename) as raw_data:
        print("the total operations's ocunt is:", len(operation_dic), "\n the longest operation length is",
              longest_operation_size)
        processed_data_np = np.empty(shape=(line_num, longest_operation_size)).astype("int32")
        labels_list = []
        for i, line in enumerate(raw_data):
            tmp_processed_data = [0 for x in range(0, longest_operation_size)]
            operation = line.replace('\n', '').replace('"', '').split(',')
            labels_list.append(operation.pop(0))
            del operation[0]
            operation_set = set(operation)
            j = 0
            for op in operation_set:
                if len(op) != 0:
                    index = operation_dic[op]
                    tmp_processed_data[j] = index
                    j += 1
            processed_data_np[i] = np.array(tmp_processed_data)

        labels_index_np = raw_labels_to_index(labels_list)

        x_train, x_test, y_train, y_test = train_test_split(processed_data_np, labels_index_np, test_size=0.01,
                                                            random_state=0)

        np.savez('F:/数据集/Kim2016/malware_dataset/malware_dataset/rnn_attention_train_test_data.npz', x_train=x_train,
                 x_test=x_test,
                 y_train=y_train, y_test=y_test)


if __name__ == '__main__':
    raw_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/malware_API_dataset - bak - selected.csv'
    op_index_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/operations_map.json'
    process_raw_data(raw_filename, op_index_filename)
