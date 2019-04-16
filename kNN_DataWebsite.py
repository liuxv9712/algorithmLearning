'''
k-近邻算法的一般流程：
收集数据：可以使用爬虫进行数据的收集，也可以使用第三方提供的免费或收费的数据。一般来讲，数据放在txt文本文件中，按照一定的格式进行存储，便于解析及处理。
准备数据：使用Python解析、预处理数据。
分析数据：可以使用很多方法对数据进行分析，例如使用Matplotlib将数据可视化。
测试算法：计算错误率。
使用算法：错误率在可接受范围内，就可以运行k-近邻算法进行分类。
'''
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from kNearestNeighbor import classify0

'''
函数说明：打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征举证
    classLabelVector - 分类Label向量
'''
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOfLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOfLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    # 返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOfLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        ##根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]=='largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector
'''
函数说明：对数据进行归一化
Parameters：
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals -数据最小值
'''
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m= dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(ranges,(m,1))
    #除以最大和最小值的差，得到归一化数据
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    #返回归一化数据结果，数据范围，最小值
    return normDataSet,ranges,minVals
'''
函数说明：分类器测试函数
Parameters:
    None
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
'''
def datingClassTest():
    #打开的文件名
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat,datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化，返回归一化后的矩阵，数据范围，数据最小值
    normMat,ranges,minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m - numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount/float(numTestVecs)*100))
'''
函数说明:通过输入一个人的三维特征,进行分类输出 
Parameters:
    无
Returns:
    无
'''
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    #打开的文件名
    filename= "datingTestSet.txt"
    #打开并处理数据
    datingDataMat,datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成numpy数组，测试集
    inArr = np.array([ffMiles,precentTats,iceCream])
    #测试集归一化
    norminArr = (inArr-minVals)/ranges
    #返回分类结果
    classifierResult = classify0(norminArr,normMat,datingLabels,3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

'''
函数说明：可视化数据
Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Returns:
    None
'''
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname = r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize = (13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i ==1:
            LabelsColors.append('black')
        if i==2:
            LabelsColors.append('orange')
        if i==3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color = LabelsColors, s =15, alpha = .5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([],[],color='black',marker='.',
                              markersize=6, label = 'didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',
                               markersize=6, label = 'smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',
                               markersize=6, label = 'largeDoses')
    #添加图例
    axs[0][0].legend(handles = [didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()
if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    print("特征矩阵是：",datingDataMat)
    print("标签向量是：",datingLabels)
    normDataSet,ranges,minVals = autoNorm(datingDataMat)
    print("归一化后的数据集：",normDataSet)
    print("数据的取值范围:",ranges)
    print("数据的最小值:",minVals)
    datingClassTest()#我们可以改变函数datingClassTest内变量hoRatio和分类器k的值，检测错误率是否随着变量值的变化而增加。依赖于分类算法、数据集和程序设置，分类器的输出结果可能有很大的不同。
    classifyPerson()
    #数据可视化
    # showdatas(datingDataMat,datingLabels)
