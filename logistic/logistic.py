"""
    logistic 回归
"""
import numpy as np
import random
def sigmoid(inX):
    """
    sigmoid func
    """
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """梯度上升"""
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() #矩阵转置
    m,n = np.shape(dataMat)
    alpha = 0.001 #向目标移动的步长
    maxCycles = 500 #迭代次数
    weights = np.ones(n, 1)
    for k in range(maxCycles):
        #计算真实类别与预测类别的差值
        h = sigmoid(dataMat*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMat.transpose()* error
    return weights

def stocGradAscent(dataMatrix, classTables, numIter = 150):
    """随机梯度上升，通过调整alpha缓解数据波动，随机抽取样本避免周期波动，加快收敛速度"""
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter): #迭代次数
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classTables[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights