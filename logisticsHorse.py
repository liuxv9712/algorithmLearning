'''
怎样解决数据缺失问题，下面给出了一些可选的做法：

使用可用特征的均值来填补缺失值；
使用特殊值来填补缺失值，如-1；
忽略有缺失值的样本；
使用相似样本的均值添补缺失值；
使用另外的机器学习算法预测缺失值。

'''
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
'''
函数说明:sigmoid函数
Parameters:
    inX - 数据
Returns:
    sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
函数说明:改进的随机梯度上升算法
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
'''
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

'''
函数说明：使用Python写的Logistic分类器做预测
Parameters:
    无
Returns:
    无
'''
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    #后加的
    testSet = [];testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
    #     if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
    #         errorCount += 1
    # errorRate = (float(errorCount)/numTestVec) *100
    # print('测试集错误率为：%.2f%%' % errorRate)
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet,trainingLabels)
    test_accurcy = classifier.score(testSet,testLabels)*100
    print('正确率：%f%%' % test_accurcy)


'''
函数说明:分类函数 
Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果
'''
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob>0.5:return 1.0
    else: return 0.0

if __name__ == '__main__':
    colicTest()
'''
错误率还是蛮高的，而且耗时1.9s，并且每次运行的错误率也是不同的，错误率高的时候可能达到40%多。为啥这样？首先，因为数据集本身有30%的数据缺失，这个是不能避免的。
另一个主要原因是，我们使用的是改进的随机梯度上升算法，因为数据集本身就很小，就几百的数据量。结论：
    当数据集较小时，我们使用梯度上升算法
    当数据集较大时，我们使用改进的随机梯度上升算法
    对应的，在Sklearn中，我们就可以根据数据情况选择优化算法，比如数据较小的时候，我们使用liblinear，数据较大时，我们使用sag和saga。
'''