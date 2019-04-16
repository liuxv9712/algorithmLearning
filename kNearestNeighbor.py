# coding=utf-8
"""
k-近邻算法步骤如下：

计算已知类别数据集中的点与当前点之间的距离；
按照距离递增次序排序；
选取与当前点距离最小的k个点；
确定前k个点所在类别的出现频率；
返回前k个点所出现频率最高的类别作为当前点的预测分类。
"""
import operator

import numpy as np

'''
Parameters:
函数说明：创建数据集
    group - 数据集
    labels - 分类标签
'''


def createDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


'''
函数说明：KNN算法，分类器
Parameters:
    inX - 用于分类的数据（测试集）
    dataSet - 用于训练的数据（训练集）
    labels - 分类标签
    k - KNN算法参数，选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
'''


def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #print(sortedClassCount)#[('动作片', 1)]
        # 返回次数最多的类别,即所要分类的类别
        return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 打印数据集
    print(group)
    print(labels)
    #k - 近邻算法没有进行数据的训练，直接使用未知的数据与已知的数据进行比较，得到结果。
    #测试集
    test = [101,20]
    #KNN分类
    test_class = classify0(test,group,labels,3)
    #打印分类结果
    print(test_class)
