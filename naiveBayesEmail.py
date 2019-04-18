'''
下面这个例子中，我们将了解朴素贝叶斯的一个最著名的应用：电子邮件垃圾过滤。首先看一下使用朴素贝叶斯对电子邮件进行分类的步骤：

收集数据：提供文本文件。
准备数据：将文本文件解析成词条向量。
分析数据：检查词条确保解析的正确性。
训练算法：使用我们之前建立的trainNB0()函数。
测试算法：使用classifyNB()，并构建一个新的测试函数来计算文档集的错误率。
使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
'''
import random
import re
import numpy as np
'''
函数说明：接受一个大字符串并将其解析为字符串列表
'''
def textParse(bigString):     #将字符串转换为字符列表
    listOfTokens = re.split(r'\W+',bigString)#将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #除了单个字母，例如大写的I，其它单词变成小写
'''
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])#创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)
'''
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters：
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量，词集模型
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]* len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:   #遍历每个词条
        if word in vocabList:   #如果词条存在于词汇表中，则置为1
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
        return returnVec    #返回文档向量
'''
函数说明：根据vocabList词汇表，构建词袋模型
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量，词袋模型
'''
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:   #遍历每个词条
        if word in inputSet:
            returnVec[vocabList.index(word)]+=1     #如果词条存在于词汇表中，则计数加一
    return returnVec   #返回词袋模型
'''
函数说明：朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vector - 非侮辱类的条件概率
    p1Vector - 侮辱类的条件概率
    pAbusive - 文档属于侮辱类的概率。abusive：辱骂的
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)      #计算训练的文档数目
    numWords = len(trainMatrix[0]) #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0 #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                      #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  #取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive  #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
'''
函数说明：朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 非侮辱类的条件概率数组
    p1Vec - 侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
'''
函数说明：测试朴素贝叶斯分类器
'''
def spamTest():
    docList = []; classList=[]; fullText = []
    for i in range(1,26):  #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i,'r').read()) #读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i,'r').read()) #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)   #创建词汇列表，不重复
    print(vocabList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!= classList[docIndex]:
            errorCount +=1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount)/len(testSet)*100))

if __name__ == '__main__':
    spamTest()