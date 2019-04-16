# coding=utf-8
'''
决策树（Decision Tree）是一种非参数的监督学习方法，决策树又称判定树，是运用于分类的一种树结构，
其中的每个内部节点代表对某一属性的一次测试，每条边代表一个测试结果，叶节点代表某个类或者类的分布。
下面使用决策树对身高体重数据进行分类：训练一个决策树分类器，输入身高和体重，分类器能判断出是瘦子还是胖子。
首先这次的训练数据一共有10个样本，每个样本有两个属性，分别为身高和体重，第三列表示类别标签：胖或瘦。该数据保存在文本文档1.txt中。
因此需要把该数据集读入，特征和类标签存放在不同变量中。此外，由于类标签是文本，需要转换为数字。
'''
import numpy as np
from seaborn.external.six import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
import graphviz
#sklearn.tree模块提供了决策树模型，用于解决分类问题和回归问题。
#本次实战内容使用的是DecisionTreeClassifier和export_graphviz，前者用于决策树构建，后者用于决策树可视化。
# 从文本文件中读取数据：
data = []
labels = []
with open('1.txt') as ifile:
    for line in ifile:
        tokens = line.strip().split(' ')#token是列表，['1.5', '50', 'thin']
        # print(tokens)
        data.append([float(tk) for tk in tokens[:-1]])#data是存放[身高，体重] 的列表
        labels.append(tokens[-1])#labels是存放[胖，瘦]类别的列表
# #程序每次从文本中读取一行，然后使用空格作为分割，将数据放到data矩阵中，然后分别将特征和类标签放到labels和y中
x = np.array(data)
# print(x)
labels = np.array(labels)
print(labels)
# numpy.array():创建多维数组
y = np.zeros(labels.shape)
print(y)#y是0数组
#由于类标签是文本，需要转换为数字
y[labels=='fat']=1
y[labels=='thin']=2
print(y)
#然后将数据随机拆分,训练数据占80%的比例，测试数据占20%的比例...random_state：是随机数的种子。种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
#格式：x_train,x_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.3, random_state=0)

#使用DecisionTreeClassifier()建立模型，并进行训练fit()
clf = tree.DecisionTreeClassifier(criterion = 'entropy')#criterion：特征选择标准，可选参数，默认是gini，可以设置为entropy。gini是基尼不纯度，是将来自集合的某种结果随机应用于某一数据项的预期误差率，是一种基于统计的思想。entropy是香农熵
                                                        #splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random。每个结点的选择策略。best参数是根据算法选择最佳的切分特征，例如gini、entropy。random随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"。
clf.fit(x_train, y_train)#fit()函数不能接收string类型的数据,所以在使用fit()前，要对string类型的数据序列化。
#测试结果是
answer = clf.predict(x_train)
print ("测试数据集是：",x_train)
print ("测试数据使用模型预测对应的类是：", answer)
print (y_train)
print (np.mean(answer == y_train))#mean()计算矩阵均值，判断预测的准不准
#测试不同特征对分类的影响权重
print (clf.feature_importances_)
#可以将决策树模型输出到模型文件中
with open("DecisionTree.dot",'w') as f:
    f = tree.export_graphviz(clf,out_file=f)
#在terminal使用命令dot -Tpdf DecisionTree.dot -o DecisionTree.pdf，可把dot文件转换成PDF文件,或者用下面的代码
# 可视化需要在GVEditli打开PDF文件才能看的到
dot_data = tree.export_graphviz(clf,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("DecisionTree.png")
graph.write_pdf("DecisionTree.pdf")