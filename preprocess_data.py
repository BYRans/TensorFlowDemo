import numpy as np

with open('F:/数据集/Kim2016/malware_dataset/malware_dataset/malware_API_dataset.csv') as raw_data:
    operations = set()
    for line in raw_data:
        operation = line.replace('\n', '').replace('"', '').split(',')
        del operation[0]
        del operation[0]
        operations = operations.union(set(operation))




filename = 'F:/数据集/Kim2016/malware_dataset/malware_dataset/operations_map.txt'
with open(filename,'w') as f:
    i = 0
    for item in operations:
        print(item, "  ", i)
        f.write(item+"  "+str(i)+"\n")
        i += 1



# stand_tmp_list = [0]*15
# stand_list = list(operations)
# stand_list.extend(stand_tmp_list)
# operations_np = np.array(stand_list).reshape(22,22,1)
# np.save('F:/数据集/mnist/tmp',operations_np)
# f = np.load('F:/数据集/mnist/tmp.npy')
# print(f)
# print(type(f))