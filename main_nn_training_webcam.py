"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/
"""

import pylab as pl
import numpy as np
import random
import nn_functions as nnf
from sklearn.datasets import load_digits
from scipy.optimize import minimize
import nn_parameters

# ==========================
# INITIALIZING PARAMETERS
# ==========================
filename_with = 'WithPerson.txt'
filename_without = 'WithoutPerson.txt'

# training_set + testing set images must be less than len(digits.images)
input_num_training = nn_parameters.input_num_training  # number of images in the training set
input_num_testing = nn_parameters.input_num_testing  # number of images in the training set

num_epochs = nn_parameters.num_epochs  # number of eras in the training phase

hidden_units = nn_parameters.hidden_units  # set the number of hidden layers units (between 64 and 10)

learning_rate = 1

# semi-static fields
_image_dimension = 64  # 8x8 pixels --> 64 pixels in total --> 64 features
_img_side = 8
_labels_num = 2  # output classes (0,1)--> 2 labels

# ==========================
# IMPORTING THE DATA
# ==========================
# Read the data from files on the disk
data_with_raw = np.loadtxt(filename_with)
data_without_raw = np.loadtxt(filename_without)

data_with_size = len(data_with_raw)
data_without_size = len(data_without_raw)

data_with = data_with_raw.reshape((data_with_size, _image_dimension))
data_without = data_without_raw.reshape((data_without_size, _image_dimension))

ones = np.ones((data_with.shape[0], 1))
zeros = np.zeros((data_without.shape[0], 1))

data_with = np.hstack((data_with, ones))  # add the 1 label (1=with person)
data_without = np.hstack((data_without, zeros))  # add the 0 label (0=without person)

dataset = np.concatenate((data_with, data_without), axis=0)  # merge the two datasets
dataset_dim = dataset.shape[0]
np.random.shuffle(dataset)  # shuffle the dataset

Xdata_raw = dataset[:, :_image_dimension]  # (830,64)

Xdata = Xdata_raw.ravel().reshape(dataset_dim, _img_side, _img_side)  # (830,8,8)
Ydata = dataset[:, _image_dimension]  # (830,1)

# ==========================
# VISUALIZING AN IMAGE
# ==========================

# show the first image of the dataset
#pl.gray()
#pl.matshow(Xdata[34])
#pl.show()
# prints the matrix which composes the image


# ==========================
# SELECTING THE DATA
# ==========================
random_numbers = np.array(random.sample(range(0, len(Xdata)), input_num_training + input_num_testing))

training_indexes = random_numbers[:input_num_training]
testing_indexes = random_numbers[input_num_training:input_num_training + input_num_testing]

# select the images and their targets for the training and the testing sets
chosen_training_images = Xdata[training_indexes]
chosen_training_outputs = Ydata[training_indexes]

chosen_testing_images = Xdata[testing_indexes]
chosen_testing_outputs = Ydata[testing_indexes]

# convert from matrices into arrays
# training sets
training_inputs = nnf.even_data(chosen_training_images, num=input_num_training, img_dim=_image_dimension)
training_outputs = nnf.even_data(chosen_training_outputs, num=input_num_training, is_input=False)
# testing sets
testing_inputs = nnf.even_data(chosen_testing_images, num=input_num_testing, img_dim=_image_dimension)
testing_outputs = nnf.even_data(chosen_testing_outputs, num=input_num_testing, is_input=False)

# ==========================
# PRE PROCESSING THE SELECTED DATA
# ==========================
X = training_inputs
# Pre-processing output data for a multi-label classification (i.e. (3)-->[0 0 0 1 0 0 0 0 0 0])
Y = nnf.onehot_y(training_outputs)

# ==========================
# INITIALIZING THE NEURAL NETWORK
# ==========================
print("Initializing the neural network (...)")
_input_units = _image_dimension  # because the images are 8x8 pixes = 64 pixels
_output_units = _labels_num  # numbers from 0 to 9

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(
    size=hidden_units * (_input_units + 1) + _output_units * (hidden_units + 1)) - 0.5) * 0.25

# compute and visualize the initial cost of the prediction with the initial (random) Theta values
initial_cost = nnf.cost(params, _input_units, hidden_units, _output_units, X, Y, learning_rate)
print("Initial cost: " + str(initial_cost))

# ==========================
# TRAINING PHASE
# ==========================
print("Training the neural network (minimizing the cost function) (...)")
# minimize the objective function
fmin = minimize(fun=nnf.back_propagate, x0=params,
                args=(_input_units, hidden_units, _output_units, X, Y, learning_rate),
                method='TNC', jac=True, options={'maxiter': num_epochs, 'disp': True})
# (fmin.x) are the Theta values which minimize the function

Theta1 = np.matrix(np.reshape(
    fmin.x[:hidden_units * (_input_units + 1)], (hidden_units, (_input_units + 1))))

Theta2 = np.matrix(np.reshape(
    fmin.x[hidden_units * (_input_units + 1):], (_output_units, (hidden_units + 1))))

# ==========================
# TRAINING ACCURACY
# ==========================
print("Calculating the training accuracy (...)")
h = nnf.forward_propagate(X, Theta1, Theta2, output_only=True)

# Returns the indices of the maximum values along an axis and creates an array
y_training_predicted = np.array(np.argmax(h, axis=1))

y_predicted_training_norm = np.squeeze(np.asarray(y_training_predicted))
y_training_norm = np.squeeze(np.asarray(training_outputs))

training_accuracy = nnf.get_accuracy(y_predicted_training_norm, y_training_norm)
print("Accuracy: " + str(training_accuracy) + "%")

# ==========================
# TESTING PHASE (UNKNOWN IMAGES)
# ==========================
print("Testing phase (...)")
h = nnf.forward_propagate(testing_inputs, Theta1, Theta2, output_only=True)

# Returns the indices of the maximum values along an axis and creates an array
y_testing_predicted = np.array(np.argmax(h, axis=1))

print("Calculating the testing accuracy (...)")
y_predicted_testing_norm = np.squeeze(np.asarray(y_testing_predicted))
y_testing_norm = np.squeeze(np.asarray(testing_outputs))

accuracy = nnf.get_accuracy(y_predicted_testing_norm, y_testing_norm)
print("Testing accuracy: " + str(accuracy) + "%")

# ==========================
# STORING WEIGHTS
# ==========================
postfix_filename = '_WebCam_epochs_' + str(num_epochs) + '_num_train_' + str(
    input_num_training) + '_' + str(_input_units) + 'x' + str(hidden_units) + 'x' + str(
    _output_units) + '.txt'

Theta1_filename = 'Theta1' + postfix_filename
Theta2_filename = 'Theta2' + postfix_filename

np.savetxt(Theta1_filename, Theta1)
np.savetxt(Theta2_filename, Theta2)

# ==========================
# RETRIEVE WEIGHTS FROM FILE
# ==========================
# Read the array from disk
Theta1_raw = np.loadtxt(Theta1_filename)
Theta2_raw = np.loadtxt(Theta2_filename)

# re-set the weights from file
new_Theta1 = Theta1_raw.reshape((hidden_units, (_input_units + 1)))
new_Theta2 = Theta2_raw.reshape((_output_units, (hidden_units + 1)))

# Check that the retrieved weights from the disk are the same
assert np.all(new_Theta1 == Theta1)
assert np.all(new_Theta2 == Theta2)
