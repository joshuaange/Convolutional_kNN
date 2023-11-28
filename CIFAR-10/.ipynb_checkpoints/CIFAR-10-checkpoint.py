import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import stats as st
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import typing_extensions

# Loading in the data
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Total data points, x pixels, y pixels, color channels")
print(x_train.shape)
print(x_test.shape)

# Transforms array into a line array
x_train = np.reshape(x_train,(x_train.shape[0],-1))
x_test = np.reshape(x_test,(x_test.shape[0],-1))

print(x_train.shape)
print(x_test.shape)

# kNN
k=1
model = NearestNeighbors(n_neighbors=k)
model.fit(x_train,y_train)
test_neighbors = model.kneighbors_graph(x_test)
print("Picking " + str(k) + " nearest neighbors, a total of " + str(x_train.shape[0]) + " training points with " + str(x_train.shape[1]) + " dimensions")

# test_nearest_neighbors looks through test_neighbors and gets the classifications of the k nearest ones
test_nearest_neighbors = np.empty((y_test.shape[0],k))
# y_pred gets the mode of the nearest neighbors
y_pred = np.empty((y_test.shape[0],1))

print("Total number of test points: " + str(x_test.shape[0]))
for i in range(x_test.shape[0]):
    if i%50 == 0:
        print("i=" + str(i))
    test_nearest_neighbors[i] = y_train[np.nonzero(test_neighbors.toarray()[i])].reshape((k))
    y_pred[i] = int(st.mode(test_nearest_neighbors[i], keepdims=True).mode)

print("Done!")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("Accuracy: " + str(np.round(100*conf_matrix.trace()/conf_matrix.sum(),2)) + "%")