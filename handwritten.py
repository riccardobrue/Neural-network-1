import pylab as pl
import numpy as np
import random
from tqdm import tqdm
from sklearn.datasets import load_digits


def derivative(x):
    return x * (1 - x)


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))  # this gives an overflow --> it is not important
    return .5 * (1 + np.tanh(.5 * x))  # same as previous --> solves the overflow problem


def preprocess_y(y, labels_num):
    processed_y = np.zeros((len(y), labels_num))
    for x in range(0, len(y)):
        processed_y[x][y[x]] = 1
    return processed_y


def postprocess_y(y):
    processed_y = np.zeros(len(y))
    for x in range(0, len(y)):
        index = np.where(y[x] == 1)[0]
        if len(index) > 1 or len(index) == 0:
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


def compute_prediction(X, syn0, syn1):
    a1 = X
    a2 = sigmoid(np.dot(a1, syn0))
    a3 = sigmoid(np.dot(a2, syn1))
    return a1, a2, a3


def revise_output(data):
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    return data


def even_data(data, num=1, img_dim=1, is_input=True):
    if is_input:
        return np.ravel(data).reshape((num, img_dim))
    else:
        return data.reshape((num, 1))


# ==========================
# IMPORTING THE DATA
# ==========================

# import the dataset
digits = load_digits()
(data, targets) = load_digits(return_X_y=True)

# ==========================
# VISUALIZING AN IMAGE
# ==========================

# show the first image of the dataset
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()
# prints the matrix which composes the image

# ==========================
# INITIALIZING PARAMETERS
# ==========================

# training_set + testing set images must be less than len(digits.images)
images_num_training_set = 1200  # number of images in the training set
images_num_testing_set = 200  # number of images in the testing set

num_eras = 2000  # number of eras in the training phase

hidden_nodes_n = 20  # set the number of hidden layers (between 64 and 10)

# semi-static fields
_image_dimension = 64  # 8x8 pixels --> 64 pixels in total --> 64 features
_labels_num = 10  # output classes (0,1,2,3,4,5,6,7,8,9)--> 10 labels

# ==========================
# SELECTING THE DATA
# ==========================

random_numbers = np.array(random.sample(range(0, len(digits.images)), images_num_training_set + images_num_testing_set))

training_indexes = random_numbers[:images_num_training_set]
testing_indexes = random_numbers[images_num_training_set:images_num_training_set + images_num_testing_set]

# select the images and their targets for the training and the testing sets
chosen_training_images = digits.images[training_indexes]
chosen_training_outputs = targets[training_indexes]

chosen_testing_images = digits.images[testing_indexes]
chosen_testing_outputs = targets[testing_indexes]

# convert from matrices into arrays
# training sets
training_input = even_data(chosen_training_images, num=images_num_training_set, img_dim=_image_dimension)
training_outputs = even_data(chosen_training_outputs, num=images_num_training_set, is_input=False)
# testing sets
testing_input = even_data(chosen_testing_images, num=images_num_testing_set, img_dim=_image_dimension)
testing_outputs = even_data(chosen_testing_outputs, num=images_num_testing_set, is_input=False)

# ==========================
# INITIALIZING THE NEURAL NETWORK
# ==========================
_input_nodes_n = _image_dimension  # because the images are 8x8 pixes = 64 pixels
_output_nodes_n = _labels_num  # numbers from 0 to 9

# input data
X = training_input

# output data
# Pre-processing output data for a multi-label classification (i.e. (3)-->[0 0 0 1 0 0 0 0 0 0])
Yin = preprocess_y(training_outputs, _labels_num)

# setting a default random each program start
# np.random.seed(1)

# building a 64-20-10 neural network (NO BIAS)
# synapses matrices (theta weights)
syn0 = 2 * np.random.random((_input_nodes_n, hidden_nodes_n)) - 1  # 64x20 matrix with random weights
syn1 = 2 * np.random.random((hidden_nodes_n, _output_nodes_n)) - 1  # 20x10 matrix with random weights

# ==========================
# TRAINING PHASE
# ==========================
print("Training the neural network")
progress_bar = tqdm(total=num_eras)
for j in range(num_eras):
    (a1, a2, a3) = compute_prediction(X, syn0, syn1)

    a3_error = Yin - a3
    a3_delta = a3_error * sigmoid(derivative(a3))

    a2_error = a3_delta.dot(syn1.T)  # transposed matrix
    a2_delta = a2_error * sigmoid(derivative(a2))

    # if (j % 200) == 0:
    #   print("Error: " + str(np.mean(np.abs(a3_error))))

    # update weights
    syn1 += a2.T.dot(a3_delta)
    syn0 += a1.T.dot(a2_delta)

    progress_bar.update(1)

progress_bar.close()

# print("Output after training")
a3 = revise_output(a3)

Yout_training = postprocess_y(a3)
# print("Original: ")
# print(chosen_training_outputs)
# print("Calculated: ")
# print(Yout_training)
accuracy = get_accuracy(chosen_training_outputs, Yout_training)
print("Training accuracy: " + str(accuracy) + "%")

# ==========================
# TESTING PHASE
# ==========================
print("Testing unknown images")
(a1, a2, a3) = compute_prediction(testing_input, syn0, syn1)
# print("Output after testing")
a3_testing = revise_output(a3)
Yout_testing = postprocess_y(a3_testing)
# print("Original (testing): ")
# print(chosen_testing_outputs)
# print("Calculated (testing): ")
# print(Yout_testing)
accuracy_testing = get_accuracy(chosen_testing_outputs, Yout_testing)
print("Testing accuracy: " + str(accuracy_testing) + "%")

# ==========================
# STORING WEIGHTS
# ==========================
syn0_filename = 'syn0_eras_' + str(num_eras) + '_num_train_' + str(images_num_training_set) + '.txt'
syn1_filename = 'syn1_eras_' + str(num_eras) + '_num_train_' + str(images_num_training_set) + '.txt'

np.savetxt(syn0_filename, syn0)
np.savetxt(syn1_filename, syn1)

# ==========================
# RETRIEVE WEIGHTS FROM FILE
# ==========================
# Read the array from disk
syn0_raw = np.loadtxt(syn0_filename)
syn1_raw = np.loadtxt(syn1_filename)

# re-set the weights from file
new_syn0 = syn0_raw.reshape((_input_nodes_n, hidden_nodes_n))
new_syn1 = syn1_raw.reshape((hidden_nodes_n, _output_nodes_n))

# Check that the retrieved weights from the disk are the same
assert np.all(new_syn0 == syn0)
assert np.all(new_syn1 == syn1)
