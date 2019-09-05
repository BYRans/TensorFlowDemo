def fit_in_vocabulary(X, voc_size):
    return [[w+1 for w in x if w < voc_size] for x in X]


a = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[1,8,9,10,11,12,13,14,15,16]]

b = fit_in_vocabulary(a,10)

print(b)