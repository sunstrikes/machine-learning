#�򵥰�knn
from numpy import *
"""
    inx: ��������
    dataset��ѵ����
    labels����ǩ���� ��ǵ�ÿ��ѵ�����������
    k��ѡ�������ھӵ���Ŀ
"""
def classify(inx, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diffMat = tile(inx, (dataset_size, 1)) - dataset #��������������size�Σ���������
    sqDiffMat = diffMat ** 2 #ŷ�Ͼ��빫ʽ
    sqDistances = sqDiffMat.sum(axis=1) #ÿ�зֱ����, �õ�����
    distances = sqDistances ** 0.5
    sortedDsitIndicies = distances.argsort() #��������
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDsitIndicies[i]]
        classCount[voteIlabel] = classCount[voteIlabel] + 1
    sortedClassCount = sorted(classCount, iteritems(),
    key = operator.imemgetter(1), reverse=True) #ѡȡ���K�����������
    return sortedClassCount[0][0]

