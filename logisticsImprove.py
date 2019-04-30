# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
'''
函数说明：加载数据
Parameters:无
Returns:
    dataMat - 数据列表 matrix:矩阵
    labelMat - 标签列表
'''
def loadDataSet():
    dataMat = []    #创建数据列表
    labelMat = []   #创建标签列表
    fr = open('testSet.txt')   #打开文件
    for line in fr.readlines():    #逐行读取
        lineArr = line.strip().split()  #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(int(lineArr[2])) #添加标签
    fr.close()  #关闭文件
    return dataMat,labelMat  #返回
'''
函数说明：sigmoid函数
Parameters:
    inX - 数据
Returns:
    sigmoid函数
'''
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
'''
函数说明：梯度上升算法gradient:梯度  ascent:上升，上坡路，n
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组
'''
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)   #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()   #转换成numpy的mat,并进行转置
    m,n = np.shape(dataMatrix)    #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01   #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500   #最大迭代次数
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #梯度上升矢量化公式
        error = labelMat - h
        weights = weights+alpha*dataMatrix.transpose()*error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array #将矩阵转换为数组，返回权重数组
'''
函数说明:改进的随机梯度上升算法
 
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
该算法第一个改进之处在于，alpha在每次迭代的时候都会调整，并且，虽然alpha会随着迭代次数不断减小，但永远不会减小到0，因为这里还存在一个常数项。必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响。如果需要处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数。
另一点值得注意的是，在降低alpha的函数中，alpha每次减少1/(j+i)，其中j是迭代次数，i是样本点的下标。
第二个改进的地方在于更新回归系数(最优参数)时，只使用一个样本点，并且选择的样本点是随机的，每次迭代不使用已经用过的样本点。这样的方法，就有效地减少了计算量，并保证了回归效果。
由于改进的随机梯度上升算法，随机选取样本点，所以每次的运行结果是不同的。
'''
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)  #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)#参数初始化
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01  #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex))) #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights)) #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  #更新回归系数
            weights_array = np.append(weights_array,weights,axis=0)#添加回归系数到数组中
            del(dataIndex[randIndex]) #删除已经使用的样本
        weights_array = weights_array.reshape(m,n) #改变维度
        return weights,weights_array #返回

'''
函数说明：绘制数据集
Parameters:
    weights - 权重参数数组
Returns：无
'''
def plotBestFit(weights):
    dataMat,labelMat = loadDataSet() #加载数据集
    dataArr = np.array(dataMat)  #转换成numpy的array数组
    n = np.shape(dataMat)[0] #数据个数
    xcord1 = [];ycord1 = [] #正样本
    xcord2 =[];ycord2 = [] #负样本
    for i in range(n):#根据数据集标签进行分类
        if int(labelMat[i]) == 1:#1为正样本
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:                     #0为负样本
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111) #添加subplot
    #scatter：散点图
    ax.scatter(xcord1,ycord1,s = 20,c = 'red',marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)    #绘制负样本
    x = np.arange(-3.0,3.0,0.1)#x轴
    y = (-weights[0] - weights[1] * x)/weights[2]#y轴
    ax.plot(x,y)#绘制图像
    plt.title('BestFit')  #绘制title
    plt.xlabel('x1'); plt.ylabel('x2')   #绘制label
    plt.show()


"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
"""
def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 10), sharex=False,sharey=False)
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)