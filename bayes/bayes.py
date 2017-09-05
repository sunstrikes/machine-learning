#!/usr/local/bin python3
#coding=utf-8

"""
朴素贝叶斯学习
"""
from numpy import *
def create_vocab_list(data_set):
    """生成一个不重复词表的list"""
    vocab_set = set([])
    for doc in data_set:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)

def word_to_vec(vocab_list, input_set):
    """将句子转化为向量"""
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] += 1
        else:
            print("the word:%s not in vocab_list" % word)
    return ret_vec
        
postlist = [['my','dog','is','dog']]
reslist = create_vocab_list(postlist)
print(reslist)
print(word_to_vec(reslist,['dog','is']))

def train(train_matrix, train_catagory):
    """分类器训练函数"""
    num_docs = len(train_matrix) #num of train data
    num_word = len(train_matrix[0])
    p_1 = sum(train_catagory)/float(num_docs)
    p0num = ones(num_word); p1num = ones(num_word)
    p0denom = 2.0; p1denom = 2.0
    for i in range(num_docs):
        if train_catagory[i] == 1:
            p1num += train_matrix[i] #向量相加
            p1denom += sum(train_matrix[i])
        else:
            p0num += train_matrix[i]
            p0denom += sum(train_matrix[i])
    p1vect = log(p1num / p1denom)
    p0vect = log(p0num / p0denom)  #求ln防止float过小，省略为0
    return p0vect, p1vect, p_1

def classify(vec2classify, p0vect, p1vect, p_1):
    """分类器分类函数"""
    p1 = sum(vec2classify * p1vect) + log(p_1)
    p0 = sum(vec2classify * p0vect) + log(1.0 - p_1)
    if p1 > p0:
        return 1
    else:
        return 0