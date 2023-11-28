import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy import stats as st
import matplotlib.pyplot as plt
from PIL import Image
import scipy

def h(x):
    # Activation function, let's just use Relu
    return np.maximum(0,x)

def convolve(kernels_r, kernels_g, kernels_b, x_pixels, y_pixels, image_data, print_names = []):
    x_pixels_convolved = x_pixels - kernels_r[0].shape[0] + 1 # new x pixels
    y_pixels_convolved = y_pixels - kernels_r[0].shape[1] + 1 # new y pixels
    channels_convolved = kernels_r.shape[0]                   # how many kernels and, hence, output channels

    # of original image:
    selected_image_r = np.empty((x_pixels,y_pixels)) # red channel
    selected_image_g = np.empty((x_pixels,y_pixels)) # blue channel
    selected_image_b = np.empty((x_pixels,y_pixels)) # green channel
    selected_image = np.empty((x_pixels,y_pixels,3))

    index = 0
    for i in range(x_pixels):
        for j in range(y_pixels):
            selected_image_r[i][j] = 255*image_data[index+0]
            selected_image_g[i][j] = 255*image_data[index+1]
            selected_image_b[i][j] = 255*image_data[index+2]
            selected_image[i][j][0] = selected_image_r[i][j]
            selected_image[i][j][1] = selected_image_g[i][j]
            selected_image[i][j][2] = selected_image_b[i][j]
            index += 3

    if len(print_names) > 0:
        image = Image.fromarray(np.array(selected_image, dtype=np.uint8), mode="RGB")
        image.save('original.png')

    convolved_image = np.empty((channels_convolved,x_pixels_convolved,y_pixels_convolved))
    # now let's cycle through each channel and apply appropriate kernels
    for c in range(channels_convolved):
        convolved_image_r = scipy.signal.convolve2d(selected_image_r,kernels_r[c],mode='valid') # red channel
        convolved_image_g = scipy.signal.convolve2d(selected_image_g,kernels_g[c],mode='valid') # blue channel
        convolved_image_b = scipy.signal.convolve2d(selected_image_b,kernels_b[c],mode='valid') # green channel
            
        for i in range(x_pixels_convolved):
            for j in range(y_pixels_convolved):
                convolved_image[c][i][j] = h(convolved_image_r[i][j] + convolved_image_g[i][j] + convolved_image_b[i][j] + 1)  

        if len(print_names) > 0:
            cm = plt.get_cmap('bwr')
            colored_image = cm(convolved_image[c]/255) # this prints out the image with the special colormap
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
            colored_image = cm(max_pooled_image[c]/255) # this prints out the image with the special colormap
            image = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(print_names[c])

    return max_pooled_image

# Load data
from sklearn.datasets import fetch_lfw_people
color = True
people = fetch_lfw_people(min_faces_per_person=3, resize=1, color=color)

x = people.data 
y = people.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

print("Total data points, x pixels, y pixels, color channels")
print(people.images.shape)
print(x_test.shape)
print(x_train.shape)

# Let's define some convolutions
x_pixels = people.images.shape[1]
y_pixels = people.images.shape[2]
windowsize = 5
number_of_kernels = 3
kernels = np.empty((number_of_kernels,3,3))
# first kernel
kernels[0] = np.array([[-1, -2, -1],
                         [0,0,0],
                         [1, 2, 1]])
# second kernel
kernels[1] = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
# third kernel
kernels[2] = (1./9.) * np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])

kernels_r = kernels
kernels_g = kernels
kernels_b = kernels

new_size = (max_pool(convolve(kernels_r, kernels_g, kernels_b, x_pixels, y_pixels, x_train[0]), windowsize).shape)
print(new_size)
# now let's do all the convolutions
x_train_new = np.empty((x_train.shape[0],(new_size[0]*new_size[1]*new_size[2])))
x_test_new = np.empty((x_test.shape[0],(new_size[0]*new_size[1]*new_size[2])))

print("Starting testing data")
for i in range(len(x_test)):
    if i % 100 ==0:
        print(i)
    convolved_image = convolve(kernels_r, kernels_g, kernels_b, x_pixels, y_pixels, x_test[i])
    x_test_new[i] = np.ravel(max_pool(convolved_image, windowsize))
print("Starting training data")
for i in range(len(x_train)):
    if i % 100 ==0:
        print(i)
    convolved_image = convolve(kernels_r, kernels_g, kernels_b, x_pixels, y_pixels, x_train[i])
    x_train_new[i] = np.ravel(max_pool(convolved_image, windowsize))

print("After convolutions:")
print(x_train_new.shape)
print(x_test_new.shape)

# let's define the folds
n_folds = 5
size_folds = int(x_train_new.shape[0]/n_folds)
x_folds = np.empty((n_folds,size_folds,x_train_new.shape[1]))
y_folds = np.empty((n_folds,size_folds))

for i in range(n_folds):
    x_folds[i] = x_train_new[size_folds*i:size_folds*(i+1)]
    y_folds[i] = y_train[size_folds*i:size_folds*(i+1)]

print("Start") # k Fold Cross Validation
k_vals = [1,2,3,5,7,9,12,15,25,50]
cv_error_vals = np.zeros((len(k_vals)))

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
                y_fold_testing = y_folds[q]
            else:
                x_fold_training[size_folds*s:size_folds*(s+1)] = x_folds[q]
                y_fold_training[size_folds*s:size_folds*(s+1)] = y_folds[q]
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