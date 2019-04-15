# coding=utf-8
'''
决策树（Decision Tree）是一种非参数的监督学习方法，决策树又称判定树，是运用于分类的一种树结构，
其中的每个内部节点代表对某一属性的一次测试，每条边代表一个测试结果，叶节点代表某个类或者类的分布。
下面使用决策树对身高体重数据进行分类：训练一个决策树分类器，输入身高和体重，分类器能判断出是瘦子还是胖子。
首先这次的训练数据一共有10个样本，每个样本有两个属性，分别为身高和体重，第三列表示类别标签：胖或瘦。该数据保存在文本文档1.txt中。
因此需要把该数据集读入，特征和类标签存放在不同变量中。此外，由于类标签是文本，需要转换为数字。
'''
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 从文本文件中读取数据：
data = []
labels = []
with open('1.txt') as ifile:
    for line in ifile:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
#程序每次从文本中读取一行，然后使用空格作为分割，将数据放到data矩阵中，然后分别将特征和类标签放到labels和y中
x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)
#由于类标签是文本，需要转换为数字
y[labels=='fat']=1
y[labels=='thin']=2
#然后将数据随机拆分,训练数据占80%的比例，测试数据占20%的比例
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#使用DecisionTreeClassifier建立模型，并进行训练
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(x_train, y_train)
#测试结果是
answer = clf.predict(x_train)
print ("测试数据集是：",x_train)
print ("测试数据使用模型预测对应的类是：", answer)
print (y_train)
print (np.mean(answer == y_train))
#测试不同特征对分类的影响权重
print (clf.feature_importances_)