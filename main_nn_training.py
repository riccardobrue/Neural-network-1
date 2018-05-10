import pylab as pl
import numpy as np
import random
import nn_functions as nnf
from tqdm import tqdm
from sklearn.datasets import load_digits
import math

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
images_num_training_set = 120  # number of images in the training set
images_num_testing_set = 20  # number of images in the testing set

num_epochs = 100  # number of eras in the training phase

hidden_nodes_n = 20  # set the number of hidden layers units (between 64 and 10)

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
training_input = nnf.even_data(chosen_training_images, num=images_num_training_set, img_dim=_image_dimension)
training_outputs = nnf.even_data(chosen_training_outputs, num=images_num_training_set, is_input=False)
# testing sets
testing_input = nnf.even_data(chosen_testing_images, num=images_num_testing_set, img_dim=_image_dimension)
testing_outputs = nnf.even_data(chosen_testing_outputs, num=images_num_testing_set, is_input=False)

# ==========================
# INITIALIZING THE NEURAL NETWORK
# ==========================
_input_nodes_n = _image_dimension  # because the images are 8x8 pixes = 64 pixels
_output_nodes_n = _labels_num  # numbers from 0 to 9

# input data
X = training_input

# output data
# Pre-processing output data for a multi-label classification (i.e. (3)-->[0 0 0 1 0 0 0 0 0 0])
Yin = nnf.preprocess_y(training_outputs, _labels_num)

# setting a default random each program start
# np.random.seed(1)

# building a 64-20-10 neural network (NO BIAS)
# synapses matrices (theta weights)
syn0 = 2 * np.random.random((_input_nodes_n, hidden_nodes_n)) - 1  # 64x20 matrix with random weights
syn1 = 2 * np.random.random((hidden_nodes_n, _output_nodes_n)) - 1  # 20x10 matrix with random weights

Theta1 = 2 * np.random.random((hidden_nodes_n, _input_nodes_n)) - 1
Theta2 = 2 * np.random.random((_output_nodes_n, hidden_nodes_n)) - 1
# ==========================
# TRAINING PHASE
# ==========================
print("Training the neural network")
progress_bar = tqdm(total=num_epochs)
m = len(Yin)
for j in range(num_epochs):
    # feed forward phase
    (a1, a2, a3) = nnf.compute_prediction_2(X, Theta1, Theta2)
    h = a3
    """
    J=(1/m)*sum(sum((-Y.*log(h))-((1-Y).*log(1-h)),2));
    """
    #cost = (1 / m) * sum(sum((-1*Yin.dot(math.log(h))) - ((1 - Yin).dot(math.log(1 - h))), 2));

    # back propagation phase
    Delta1 = 0
    Delta2 = 0
    for t in range(m):
        q1 = (X[t, :]).T  # 64x1

        z2 = Theta1.dot(q1)  # 20x1
        q2 = nnf.sigmoid(z2)  # 20x1

        z3 = Theta2.dot(q2)  # 10x1
        q3 = nnf.sigmoid(z3)  # 10x1

        d3 = q3 - (Yin[t, :]).T  # 10x1
        d2 = (Theta2.T.dot(d3)) * (nnf.derivative(z2))  # 20x1

        Delta1 += (np.reshape(d2, (20, 1)).dot(np.reshape(q1, (1, 64))))
        Delta2 += (np.reshape(d3, (10, 1)).dot(np.reshape(q2, (1, 20))))

    # if (j % 1000) == 0:
    # print("Error: " + str(np.mean(np.abs(a3_error))))

    # update weights
    Theta1 = (1 / m) * Delta1
    Theta2 = (1 / m) * Delta2

    progress_bar.update(1)

progress_bar.close()

# print("Output after training")
a3_training = nnf.revise_output(a3)
print(a3_training)

Yout_training = nnf.postprocess_y(a3_training)
# print("Original: ")
# print(chosen_training_outputs)
# print("Calculated: ")
# print(Yout_training)
accuracy = nnf.get_accuracy(chosen_training_outputs, Yout_training)
print("Training accuracy: " + str(accuracy) + "%")

# ==========================
# TESTING PHASE
# ==========================
print("Testing unknown images")
(a1, a2, a3) = nnf.compute_prediction_2(testing_input, Theta1, Theta2)
# print("Output after testing")
a3_testing = nnf.revise_output(a3)
Yout_testing = nnf.postprocess_y(a3_testing)
# print("Original (testing): ")
# print(chosen_testing_outputs)
# print("Calculated (testing): ")
# print(Yout_testing)
accuracy_testing = nnf.get_accuracy(chosen_testing_outputs, Yout_testing)
print("Testing accuracy: " + str(accuracy_testing) + "%")

"""
# ==========================
# STORING WEIGHTS
# ==========================
postfix_filename = '_epochs_' + str(num_epochs) + '_num_train_' + str(
    images_num_training_set) + '_' + str(_input_nodes_n) + 'x' + str(hidden_nodes_n) + 'x' + str(
    _output_nodes_n) + '.txt'

syn0_filename = 'syn0' + postfix_filename
syn1_filename = 'syn1' + postfix_filename

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
"""
