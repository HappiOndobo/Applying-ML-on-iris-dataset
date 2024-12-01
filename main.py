import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import sklearn
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from laboratory_ML.test import iris_dataset

iris_dataset = load_iris()
print("Keys of iris_dataset: {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("shape of data: {}".format(iris_dataset['data'].shape))
print ( "First five rows of data: \n{}" . format ( iris_dataset [ 'data' ][ : 5]))

X_train, X_test,  y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()