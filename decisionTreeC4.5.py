# -*- coding: utf-8 -*-
from math import log
import operator
import pickle

def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：数据集的香农熵
    描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
    """
    numEntries = len(dataSet)#返回数据集的行数
    print("样本总数：", numEntries)
    labelCounts = {}#创建数据字典，记录每一类标签的数量
    # 定义特征向量featVec
    for featVec in dataSet:
        currentLabel = featVec[-1]#最后一列是类别标签
        if currentLabel not in labelCounts.keys(): #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  #Label计数，标签currentLabel出现的次数
        print("字典labelCounts的值即标签数量：" + str(labelCounts))
    #使用所有类标签发生的频率计算类别出现的概率，计算香农熵（shang）
    shannonEnt = 0.0     #经验熵(香农熵)
    for key in labelCounts:   #计算香农熵
        prob = float(labelCounts[key]) / numEntries#每一个类别标签出现的概率
        print(str(key) + "类别的概率：" + str(prob))
        # print(prob * log(prob, 2))
        shannonEnt -= prob * log(prob, 2)  #利用公式计算
        print("熵值：" + str(shannonEnt))
    return shannonEnt  #返回经验熵(香农熵)


def splitDataSet(dataSet, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    retDataSet = [] #创建返回的数据集列表
    for featVec in dataSet:#遍历数据集
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]#去掉axis特征
            reduceFeatVec.extend(featVec[axis + 1:])#将符合条件的添加到返回的数据集
            retDataSet.append(reduceFeatVec)
    return retDataSet  #返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1#特征数量
    baseEntropy = calcShannonEnt(dataSet)#计算数据集的香农熵
    bestInfoGainRatio = 0.0#信息增益
    bestFeature = -1 #最优特征的索引值
    for i in range(numFeatures):  #遍历所有特征
        featList = [example[i] for example in dataSet] #获取dataSet的第i个所有特征
        uniqueVals = set(featList) #创建set集合{},元素不可重复
        newEntropy = 0.0  #计算信息增益
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))   #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)   #根据公式计算经验条件熵
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy  #信息增益
        print("信息增益是：", infoGain)
        if (splitInfo == 0):  # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio #更新信息增益，找到最大的信息增益
            bestFeature = i  #记录信息增益最大的特征的索引值
    return bestFeature   #返回信息增益最大的特征的索引值


def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类。统计classList中出现此处最多的元素(类标签)
    """
    classCount = {}
    for vote in classList:  #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)#根据字典的值降序排序
    return sortedClassCount[0][0] #返回classList中出现次数最多的元素


def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]  #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):  #如果类别完全相同则停止继续划分
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel = labels[bestFeat] #最优特征的标签
    myTree = {bestFeatLabel: {}}  #根据最优特征的标签生成树
    del (labels[bestFeat])   #删除已经使用特征标签

    featValues = [example[bestFeat] for example in dataSet]  # 得到列表包括节点所有的属性值
    uniqueVals = set(featValues)#去掉重复的属性值
    for value in uniqueVals:       #遍历特征，创建决策树。
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


def storeTree(inputTree, filename):
    """
    输入：决策树，保存文件路径
    输出：
    描述：保存决策树到文件
    为了节省计算时间，最好能够在每次执行分类时调用已经构造好的决策树。为了解决这个问题，需要使用Python模块pickle序列化对象。序列化对象可以在磁盘上保存对象，并在需要的时候读取出来。
    """
    with open(filename, 'w')as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    输入：文件路径名
    输出：决策树
    描述：从文件读取决策树
    将决策树存储完这个二进制文件，然后下次使用的话，怎么用呢？
很简单使用pickle.load进行载入即可，编写代码如下：
    """
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels


def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet


def main():
    dataSet, labels = createDataSet()
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels
    desicionTree = createTree(dataSet, labels_tmp)
    # storeTree(desicionTree, 'classifierStorage.txt')
    # desicionTree = grabTree('classifierStorage.txt')
    print('desicionTree:\n', desicionTree)
    # treePlotter.createPlot(desicionTree)
    testSet = createTestSet()
    print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))


if __name__ == '__main__':
    main()