import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import sklearn
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from laboratory_ML.test import iris_dataset

iris_dataset = load_iris()
"""print("Keys of iris_dataset: {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("shape of data: {}".format(iris_dataset['data'].shape))
print ( "First five rows of data: \n{}" . format ( iris_dataset [ 'data' ][ : 5]))
"""
X_train, X_test,  y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_test shape: {}".format(X_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
#print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
#print("Prediction: {}".format(prediction))
#print("Predicted target name: {}".format(iris_dataset[ 'target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test before classification:\n {}".format(y_test))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

















