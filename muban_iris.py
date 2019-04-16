'''模板'''
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
#1 载入sciki-learn的自带数据，有决策树拟合，得到模型
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
#2 将模型存入dot文件iris.dot
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
#3 可视化方法，
# 第一种是用graphviz的dot命令生成决策树的可视化文件
#dot -Tpdf iris.dot -o iris.pdf
#第二种方法是用pydotplus生成iris.pdf。
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
#三种办法是个人比较推荐的做法，因为这样可以直接把图产生在ipython的notebook。
from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())