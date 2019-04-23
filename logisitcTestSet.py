# -*- coding: utf-8 -*-
'''
Logistic回归的一般过程：

收集数据：采用任意方法收集数据。
准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
分析数据：采用任意方法对数据进行分析。
训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
测试算法：一旦训练步骤完成，分类将会很快。
使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数，就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；在这之后，我们就可以在输出的类别上做一些其他分析工作。
'''

import matplotlib.pyplot as plt
import numpy as np

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
    alpha = 0.001   #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500   #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #梯度上升矢量化公式
        error = labelMat - h
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights.getA()  #将矩阵转换为数组，返回权重数组

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
    ax.scatter(xcord1,ycord1,s = 20,c = 'red',marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)    #绘制负样本
    x = np.arange(-3.0,3.0,0.1)#x轴
    y = (-weights[0] - weights[1] * x)/weights[2]#y轴
    ax.plot(x,y)#绘制图像
    plt.title('BestFit')  #绘制title
    plt.xlabel('x1'); plt.ylabel('x2')   #绘制label
    plt.show()

if __name__ == '__main__':
    #数据可视化
    # plotDataSet()
    dataMat , labelMat = loadDataSet()
    print(gradAscent(dataMat,labelMat))#求解出回归系数[w0,w1,w2]，就可以确定不同类别数据之间的分隔线，画出决策边界。
    weights = gradAscent(dataMat,labelMat)
    plotBestFit(weights)