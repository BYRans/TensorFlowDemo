import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from keras.utils import np_utils
import re


def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def create_oprations_set(raw_filename, op_index_filename):
    with open(raw_filename) as raw_data:
        operations = set()
        for line in raw_data:
            operation = line.replace('\n', '').split(' ')
            operations = operations.union(set(operation))

    operations_list = list(operations)
    map_index = range(len(operations))
    ziped_op_index = zip(operations_list, map_index)

    operations_dic = {k: v + 1 for k, v in ziped_op_index}  # V+1 for no operation index is 0

    with open(op_index_filename, 'w') as json_file:
        json.dump(operations_dic, json_file, ensure_ascii=False)
    print("operations index dictionary create success! Dic file saved in ", op_index_filename)
    print("the operations's count is:", len(operations))


def raw_labels_to_index(raw_lable_filename):
    raw_lables_list = []

    with open(raw_lable_filename) as raw_data:
        for line in raw_data:
            raw_lables_list.append(line.replace('\n', '').strip())

    lables_set_index = {}
    for lable in raw_lables_list:
        if lable not in lables_set_index:
            lables_set_index[lable] = len(lables_set_index)
            print(lable, "--", lables_set_index[lable])
    print("the lables's count is:", len(lables_set_index))
    lables_index_np = np.zeros(len(raw_lables_list))
    for i, lable in enumerate(raw_lables_list):
        lables_index_np[i] = lables_set_index.get(lable)
    return lables_index_np


def load_npz_data(file_path):
    f = np.load(file_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def process_raw_data_4_attention(raw_api_filename, raw_lable_filename, op_index_filename,
                                 attention_train_data, no_repet_api_filename):
    apis = []
    apis_str = ""
    with open(raw_api_filename) as raw_data:
        i = 0
        for line in raw_data:
            operations = line.replace('\n', '').split(' ')
            last_operation = ""
            tmp_api = []
            for op in operations:
                op = str(op)
                if op == last_operation:
                    continue
                else:
                    tmp_api.append(op)
                    last_operation = op
                last_operation = op
            tmp_api_str = " ".join(tmp_api)
            apis.append(tmp_api_str)
            i = i + 1
            if i % 100 == 0:
                print(i)
        apis_str = "\n".join(apis)

    with open(no_repet_api_filename, 'w')as g:
        g.write(apis_str)
        g.close()


def main():
    raw_api_filename = 'F:/数据集/ocatak/mal-api-2019/all_analysis_data.txt'
    raw_lable_filename = 'F:/数据集/ocatak/labels.csv'
    op_index_filename = 'F:/数据集/ocatak/operations_map_final.json'
    attention_train_data = "F:/数据集/ocatak/attention_train_test_data_final.npz"
    no_repet_api_filename = 'F:/数据集/ocatak/mal-api-2019/no_repet_api_data.txt'

    # create_oprations_set(raw_api_filename, op_index_filename)
    process_raw_data_4_attention(raw_api_filename, raw_lable_filename, op_index_filename,
                                 attention_train_data, no_repet_api_filename)
    process_raw_data_4_attention('F:/数据集/ocatak/mal-api-2019/sample_analysis_data.csv', raw_lable_filename, op_index_filename, attention_train_data,
                                 'F:/数据集/ocatak/mal-api-2019/sample_no_repet_api_data.txt')
    # (X_Train, y_Train), (X_Test, y_Test) = load_npz_data("F:/数据集/Kim2016/malware_dataset/malware_dataset/train_test_data_final.npz")


if __name__ == "__main__":
    main()
