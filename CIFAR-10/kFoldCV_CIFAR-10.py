import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import stats as st
import matplotlib.pyplot as plt
from PIL import Image
import scipy

# Loading in the data
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Total data points, x pixels, y pixels, color channels")
print(x_train.shape)
print(x_test.shape)

# Transforms array into a line array
x_train_new = np.reshape(x_train,(x_train.shape[0],-1))
x_test_new = np.reshape(x_test,(x_test.shape[0],-1))

print(x_train.shape)
print(x_test.shape)

# let's define the folds for k-Fold cross validation
n_folds = 3
size_folds = int(x_train_new.shape[0]/n_folds) # how many elements in each fold, and separate them in x_folds and y_folds
x_folds = np.empty((n_folds,size_folds,x_train_new.shape[1]))
y_folds = np.empty((n_folds,size_folds,1))

for i in range(n_folds):
    x_folds[i] = x_train_new[size_folds*i:size_folds*(i+1)]
    y_folds[i] = y_train[size_folds*i:size_folds*(i+1)].reshape((-1))

print("Start") # k Fold Cross Validation
k_vals = [1,2,3,5,7,9,12,15,25,50]
cv_error_vals = np.zeros((len(k_vals))) # here's an array of cv error

# kNN
for i in range(len(k_vals)):
    k = k_vals[i]
    print("k value of " + str(k))
    
    for j in range(n_folds):

        # let's set up training and testing data
        s = 0
        x_fold_training = np.empty((size_folds*(n_folds-1),x_train_new.shape[1]))
        y_fold_training = np.empty((size_folds*(n_folds-1)))
        x_fold_testing = np.empty((size_folds,x_train_new.shape[1]))
        y_fold_testing = np.empty((size_folds))
        
        for q in range(n_folds):
            if j==q:
                x_fold_testing = x_folds[q]
                y_fold_testing = y_folds[q].reshape((-1))
            else:
                x_fold_training[size_folds*s:size_folds*(s+1)] = x_folds[q]
                y_fold_training[size_folds*s:size_folds*(s+1)] = y_folds[q].reshape((-1))
                s += 1

        model = NearestNeighbors(n_neighbors=k)
        model.fit(x_fold_training,y_fold_training)
        test_neighbors = model.kneighbors_graph(x_fold_testing)
        
        # test_nearest_neighbors looks through test_neighbors and gets the classifications of the k nearest ones
        test_nearest_neighbors = np.empty((y_fold_testing.shape[0],k))
        # y_pred gets the mode of the nearest neighbors
        y_pred = np.empty((y_fold_testing.shape[0],1))
        
        print("Total number of test points: " + str(x_fold_testing.shape[0]))
        for w in range(x_fold_testing.shape[0]):
            if w%100 == 0:
                print("i=" + str(w))
            test_nearest_neighbors[w] = y_fold_training[np.nonzero(test_neighbors.toarray()[w])].reshape((k))
            y_pred[w] = int(st.mode(test_nearest_neighbors[w], keepdims=True).mode)
        
        conf_matrix = confusion_matrix(y_fold_testing, y_pred)
        print("k error:" + str(1 - conf_matrix.trace()/conf_matrix.sum()))
        cv_error_vals[i] += float((1. - conf_matrix.trace()/conf_matrix.sum())/n_folds) # adding it to the cv error array

print(k_vals)
print(cv_error_vals)