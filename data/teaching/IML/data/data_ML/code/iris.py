import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
for xx in range(4):
    for yy in range(4):
        if yy > xx:
            print(xx, yy)
            plt.xlabel(labels[xx])
            plt.ylabel(labels[yy])            
            plt.scatter(iris.data[y==0][:, xx], iris.data[y==0][:,yy])
            plt.scatter(iris.data[y==1][:, xx], iris.data[y==1][:,yy])
            plt.scatter(iris.data[y==2][:, xx], iris.data[y==2][:,yy])
            plt.show()

