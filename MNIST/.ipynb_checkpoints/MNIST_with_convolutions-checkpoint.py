import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import stats as st
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import typing_extensions

def h(x):
    # Activation function, let's just use Relu
    return np.maximum(0,x)

def convolve(kernels, x_pixels, y_pixels, image_data, print_names = []):
    x_pixels_convolved = x_pixels - kernels[0].shape[0] + 1 # new x pixels
    y_pixels_convolved = y_pixels - kernels[0].shape[1] + 1 # new y pixels
    channels_convolved = kernels.shape[0]                   # how many kernels and, hence, output channels

    # of original image:
    selected_image = image_data

    if len(print_names) > 0:
        image = Image.fromarray((selected_image).astype(np.uint8)).save("original.png")

    convolved_image = np.empty((channels_convolved,x_pixels_convolved,y_pixels_convolved))
    # now let's cycle through each channel and apply appropriate kernels
    for c in range(channels_convolved):
        convolved_image_raw = scipy.signal.convolve2d(selected_image,kernels[c],mode='valid')
            
        for i in range(x_pixels_convolved):
            for j in range(y_pixels_convolved):
                convolved_image[c][i][j] = h(convolved_image_raw[i][j])  

        if len(print_names) > 0:
            cm = plt.get_cmap('bwr')
            colored_image = cm(convolved_image[c]/255)
            image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(print_names[c])
        
    return convolved_image

def max_pool(input_image, windowsize, print_names = []):
    x_pixels = int(input_image[0].shape[0]/windowsize) # new x pixels
    y_pixels = int(input_image[0].shape[1]/windowsize) # new y pixels
    channels = int(input_image.shape[0])     # channels
    
    max_pooled_image = np.zeros((channels,x_pixels,y_pixels))
    # now let's cycle through each channel and max pool
    for c in range(channels):
        for i in range(x_pixels):
            for j in range(y_pixels):
                sum = np.zeros((windowsize,windowsize))
                for i2 in range(windowsize):
                    for j2 in range(windowsize):
                        sum[i2][j2] = input_image[c][windowsize*i + i2][windowsize*j + j2]
                max_pooled_image[c][i][j] = np.max(sum)

        if len(print_names) > 0:
            cm = plt.get_cmap('bwr')
            colored_image = cm(max_pooled_image[c]/255)
            image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(print_names[c])

    return max_pooled_image
    
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Total data points, x pixels, y pixels, color channels")
print(x_test.shape)

# Let's define some convolutions
x_pixels = 28
y_pixels = 28
windowsize = 5
number_of_kernels = 3
kernels = np.empty((number_of_kernels,3,3))
# first kernel
kernels[0] = np.array([[1,1,1],
                         [0,0,0],
                         [-1, -1, -1]])
# second kernel
kernels[1] = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
# third kernel
kernels[2] = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])

new_size = (max_pool(convolve(kernels, x_pixels, y_pixels, x_train[0]), windowsize).shape)
print(new_size)

x_train_new = np.empty((x_train.shape[0],(new_size[0]*new_size[1]*new_size[2])))
x_test_new = np.empty((x_test.shape[0],(new_size[0]*new_size[1]*new_size[2])))

print("Starting testing data")
for i in range(len(x_test)):
    if i % 100 ==0:
        print(i)
    convolved_image = convolve(kernels, x_pixels, y_pixels, x_test[i])
    x_test_new[i] = np.ravel(max_pool(convolved_image, windowsize))
print("Starting training data")
for i in range(len(x_train)):
    if i % 100 ==0:
        print(i)
    convolved_image = convolve(kernels, x_pixels, y_pixels, x_train[i])
    x_train_new[i] = np.ravel(max_pool(convolved_image, windowsize))

print("After convolutions:")
print(x_train_new.shape)
print(x_test_new.shape)

# kNN
print("Start")
k=10
model = NearestNeighbors(n_neighbors=k)
model.fit(x_train_new,y_train)
test_neighbors = model.kneighbors_graph(x_test_new)
print("Picking " + str(k) + " nearest neighbors, a total of " + str(x_train_new.shape[0]) + " training points with " + str(x_train_new.shape[1]) + " dimensions")

# test_nearest_neighbors looks through test_neighbors and gets the classifications of the k nearest ones
test_nearest_neighbors = np.empty((y_test.shape[0],k))
# y_pred gets the mode of the nearest neighbors
y_pred = np.empty((y_test.shape[0],1))

print("Total number of test points: " + str(x_test.shape[0]))
for i in range(x_test_new.shape[0]):
    if i%100 == 0:
        print("i=" + str(i))
    test_nearest_neighbors[i] = y_train[np.nonzero(test_neighbors.toarray()[i])].reshape((k))
    y_pred[i] = int(st.mode(test_nearest_neighbors[i], keepdims=True).mode)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("Accuracy: " + str(np.round(100*conf_matrix.trace()/conf_matrix.sum(),2)) + "%")
print("Random: " + str(np.round(1/conf_matrix.shape[0],5)) + "%")
print(x_train.shape)