import numpy as np
import json


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


def process_raw_data(raw_filename, op_index_filename):
    np.set_printoptions(threshold=np.inf)

    with open(op_index_filename, 'r') as fileR:
        operation_dic = json.load(fileR)
        fileR.close()

    with open(raw_filename) as raw_data:
        line_num = len(raw_data.readlines())

    with open(raw_filename) as raw_data:
        processed_data = np.empty(shape=(line_num, len(operation_dic))).astype("int32")
        labels_list = []
        i = 0
        for line in raw_data:
            tmp_processed_data = [0 for x in range(0, len(operation_dic))]
            operation = line.replace('\n', '').replace('"', '').split(',')
            labels_list.append(operation.pop(0))
            mal_name = operation.pop(0)
            operation_set = set(operation)
            for op in operation_set:
                if len(op) != 0:
                    index = operation_dic[op]
                    tmp_processed_data[index] = 1
            processed_data[i] = np.array(tmp_processed_data)
            i += 1

        f = open('F:/数据集/Kim2016/malware_dataset/malware_dataset/labels.txt',"w")
        f.write(str(set(labels_list)))

def main():
    raw_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/malware_API_dataset.csv'
    op_index_filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/operations_map.json'

    # create_oprations_set(raw_filename, op_index_filename)
    process_raw_data(raw_filename, op_index_filename)


if __name__ == "__main__":
    main()
