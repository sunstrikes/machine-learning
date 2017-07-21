#简单版knn
from numpy import *
"""
    inx: 输入向量
    dataset：训练集
    labels：标签向量 标记的每个训练样本的类别
    k：选择的最近邻居的数目
"""
def classify(inx, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diffMat = tile(inx, (dataset_size, 1)) - dataset #将输入向量复制size次，与矩阵相减
    sqDiffMat = diffMat ** 2 #欧氏距离公式
    sqDistances = sqDiffMat.sum(axis=1) #每行分别求和, 得到向量
    distances = sqDistances ** 0.5
    sortedDsitIndicies = distances.argsort() #距离排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDsitIndicies[i]]
        classCount[voteIlabel] = classCount[voteIlabel] + 1
    sortedClassCount = sorted(classCount, iteritems(),
    key = operator.imemgetter(1), reverse=True) #选取最近K个的最多的类别
    return sortedClassCount[0][0]

