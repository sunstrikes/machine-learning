from math import log
del calShannon(dataset):
    #dataset: 二维数组
    #信息熵：-p(x)*log2(p(x))求和
    data_len = len(dataset)
    counts = {}
    for i in dataset:
        cur_key = i[-1]
        if cur_key not in counts.keys():
            counts[cur_key] = 0
        counts[cur_key]+=1
    res = 0
    for key in counts:
        prob = float(counts[key]/data_len); //p(x)
        res -= prob * log(prob, 2)
    return res


