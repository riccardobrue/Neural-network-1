import scipy
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn.datasets import load_digits


def derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # this gives an overflow --> it is not important


def preprocess_y(y, labels_num):
    processed_y = np.zeros((len(y), labels_num))
    for x in range(0, len(y)):
        processed_y[x][y[x]] = 1
    return processed_y


def postprocess_y(y):
    processed_y = np.zeros(len(y))
    for x in range(0, len(y)):
        index = np.where(y[x] == 1)[0]
        if len(index) > 1:
            processed_y[x] = -1
        else:
            processed_y[x] = index
    return processed_y.astype(np.int32)


def get_accuracy(y1, y2):
    size = len(y1)
    count = 0
    for x in range(0, size):
        if y1[x] == y2[x]:
            count += 1
    return (count * 100) / size


# import the dataset
digits = load_digits()
(data, targets) = load_digits(return_X_y=True)

# show the first image of the dataset
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()
# prints the matrix which composes the image

# get the training set of 50 images
number_of_images = 50
image_dimension = 64
labels_num = 10

# select the first 50 images and their targets
chosen_images = digits.images[:number_of_images]
chosen_outputs = targets[:number_of_images]

# convert from matrices into arrays
training_set = np.ravel(chosen_images).reshape((number_of_images, image_dimension))
training_outputs = chosen_outputs.reshape((number_of_images, 1))

# starting the neural network phase
input_nodes_n = image_dimension  # because the images are 8x8 pixes = 64 pixels
hidden_nodes_n = 20  # just picked a value
output_nodes_n = labels_num  # numbers from 0 to 9

# input data
X = training_set

# output data
# Pre-processing output data for a multi-label classification (i.e. (3)-->[0 0 0 1 0 0 0 0 0 0])
Y = preprocess_y(training_outputs, labels_num)

# setting a default random each program start
np.random.seed(1)

# building a 64-20-10 neural network (NO BIAS)
# synapses matrices (theta weights)
syn0 = 2 * np.random.random((64, 20)) - 1  # 64x20 matrix with random weights
syn1 = 2 * np.random.random((20, 10)) - 1  # 20x10 matrix with random weights


# training step
num_eras = 20000
for j in range(num_eras):
    a1 = X
    a2 = sigmoid(np.dot(a1, syn0))
    a3 = sigmoid(np.dot(a2, syn1))

    a3_error = Y - a3
    a3_delta = a3_error * sigmoid(derivative(a3))

    a2_error = a3_delta.dot(syn1.T)  # transposed matrix
    a2_delta = a2_error * sigmoid(derivative(a2))

    # if (j % 20) == 0:
    # print("Error: " + str(np.mean(np.abs(a3_error))))

    # update weights
    syn1 += a2.T.dot(a3_delta)
    syn0 += a1.T.dot(a2_delta)

print("Output after training")
a3[a3 >= 0.5] = 1
a3[a3 < 0.5] = 0

Yout = postprocess_y(a3)
print("Original: ")
print(chosen_outputs)
print("Calculated: ")
print(Yout)
accuracy = get_accuracy(chosen_outputs, Yout)
print("Accuracy: " + str(accuracy) + "%")
