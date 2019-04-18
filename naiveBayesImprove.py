'''
拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。
下溢出，这是由于太多很小的数相乘造成的,两个小数相乘，越乘越小，这样就造成了下溢出。通过求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失。

因此我们可以对上篇文章的trainNB0(trainMatrix, trainCategory)函数进行更改'''
# -*- coding: UTF-8 -*-
from functools import reduce
import numpy as np

'''
函数说明：创建实验样本
Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签分量
'''
def loadDataSet():
    postingList=[['my','dog','has','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                 ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
'''
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量，词集模型
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #创建一个其中所含元素都为0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1   #如果词条存在于词汇表中，则置1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  #返回文档向量
'''
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇条
Parameters：
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])   #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)   #取并集
    return list(vocabSet)
'''
函数说明：朴素贝叶斯分类器训练函数
Parameters：
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 非侮辱类的条件概率数组
    p1Vect - 侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)     #计算训练的文档数目
    numWords = len(trainMatrix[0])     #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #文档属于侮辱类的概率
    p0Num = np.ones(numWords)  #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p1Num = np.ones(numWords)  #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0 #分母初始化为2,拉普拉斯平滑
    p1Denom = 2.0 #分母初始化为2，拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:   #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else: #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)   #取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)   #取对数，防止下溢出
    return p0Vect,p1Vect,pAbusive   #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
'''
函数说明：朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
'''
#这样我们得到的结果就没有问题了，不存在0概率。,classifyNB（）也要修改
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1 = reduce(lambda x,y:x*y,vec2Classify * p1Vec) * pClass1
    # p0 = reduce(lambda x,y:x*y,vec2Classify*p0Vec) * (1.0-pClass1)
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)   #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1.0-pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1>p0:
        return 1
    else:
        return 0
#这样，我们的朴素贝叶斯分类器就改进完毕了。
'''
函数说明：测试朴素贝叶斯分类器
'''
def testingNB():
    listPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')

if __name__ == '__main__':
    postingList, classVec = loadDataSet()#postingList是存放词条列表
    # for each in postingList:
    #     print(each)
    # print(classVec)#classVec是存放每个词条的所属类别，1代表侮辱类 ，0代表非侮辱类。
    print('postingList:\n', postingList)#postingList是原始的词条列表
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)#myVocabList是所有单词出现的集合，没有重复的元素。
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print('trainMat:\n',trainMat)#将词条向量化的，一个单词在词汇表中出现过一次，那么就在相应位置记作1，如果没有出现就在相应位置记作0。trainMat是所有的词条向量组成的列表。
    p0V, p1V, pAb = trainNB0(trainMat,classVec)
    print('属于非侮辱类词汇的概率是p0V:\n',p0V)
    print('属于侮辱类词汇的概率是p1V:\n',p1V)
    print('classVec:\n',classVec)
    print('所有侮辱类的样本占所有样本的概率是pAb:\n',pAb)
    testingNB()