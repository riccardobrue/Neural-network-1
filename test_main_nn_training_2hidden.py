import pylab as pl
import numpy as np
import random
import nn_functions as nnf
from tqdm import tqdm
from sklearn.datasets import load_digits

# ==========================
# IMPORTING THE DATA
# ==========================

# import the dataset
digits = load_digits()
(data, targets) = load_digits(return_X_y=True)

# ==========================
# INITIALIZING PARAMETERS
# ==========================

# training_set + testing set images must be less than len(digits.images)
images_num_training_set = 1500  # number of images in the training set
images_num_testing_set = 200  # number of images in the testing set

num_epochs = 4000  # number of eras in the training phase

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

# building a 64-20-10 neural network (WITHOUT BIAS)
# synapses matrices (theta weights)

syn0 = 2 * np.random.random((_input_nodes_n, hidden_nodes_n)) - 1  # 64x30 matrix with random weights
syn1 = 2 * np.random.random((hidden_nodes_n, hidden_nodes_n)) - 1  # 30x30 matrix with random
syn2 = 2 * np.random.random((hidden_nodes_n, _output_nodes_n)) - 1  # 30x10 matrix with random weights

# ==========================
# TRAINING PHASE
# ==========================
print("Training the neural network")
progress_bar = tqdm(total=num_epochs)
for j in range(num_epochs):
    (a1, a2, a3, a4) = nnf.compute_prediction_3(X, syn0, syn1, syn2)

    a4_error = Yin - a4
    a4_delta = a4_error * nnf.derivative(a4)  # slope of the sigmoid at the values in a4

    a3_error = a4_delta.dot(syn2.T)  # transposed matrix
    a3_delta = a3_error * nnf.derivative(a3)  # slope of the sigmoid at the values in a3

    a2_error = a3_delta.dot(syn1.T)  # transposed matrix
    a2_delta = a2_error * nnf.derivative(a2)  # slope of the sigmoid at the values in a2



    if (j % 200) == 0:
        print("Error: " + str(np.mean(np.abs(a4_error))))

    # update weights
    syn2 += a3.T.dot(a4_delta)
    syn1 += a2.T.dot(a3_delta)
    syn0 += a1.T.dot(a2_delta)

    # update the progress bar
    progress_bar.update(1)

progress_bar.close()

# print("Output after training")
a4_training = nnf.revise_output(a4)

Yout_training = nnf.postprocess_y(a4_training)
accuracy = nnf.get_accuracy(chosen_training_outputs, Yout_training)
print("Training accuracy: " + str(accuracy) + "%")

# ==========================
# TESTING PHASE
# ==========================
print("Testing unknown images")
(a1, a2, a3, a4) = nnf.compute_prediction_3(testing_input, syn0, syn1, syn2)
a4_testing = nnf.revise_output(a4)
Yout_testing = nnf.postprocess_y(a4_testing)
accuracy_testing = nnf.get_accuracy(chosen_testing_outputs, Yout_testing)
print("Testing accuracy: " + str(accuracy_testing) + "%")

"""
# ==========================
# STORING WEIGHTS
# ==========================
postfix_filename = '_epochs_' + str(num_epochs) + '_num_train_' + str(
    images_num_training_set) + '_' + str(_input_nodes_n) + 'x' + str(hidden_nodes_n) + 'x' + str(
    hidden_nodes_n) + 'x' + str(
    _output_nodes_n) + '.txt'

syn0_filename = 'syn0' + postfix_filename
syn1_filename = 'syn1' + postfix_filename
syn2_filename = 'syn2' + postfix_filename

np.savetxt(syn0_filename, syn0)
np.savetxt(syn1_filename, syn1)
np.savetxt(syn2_filename, syn2)

# ==========================
# RETRIEVE WEIGHTS FROM FILE
# ==========================
# Read the array from disk
syn0_raw = np.loadtxt(syn0_filename)
syn1_raw = np.loadtxt(syn1_filename)
syn2_raw = np.loadtxt(syn2_filename)

# re-set the weights from file
new_syn0 = syn0_raw.reshape((_input_nodes_n, hidden_nodes_n))
new_syn1 = syn1_raw.reshape((hidden_nodes_n, hidden_nodes_n))
new_syn2 = syn2_raw.reshape((hidden_nodes_n, _output_nodes_n))

# Check that the retrieved weights from the disk are the same
assert np.all(new_syn0 == syn0)
assert np.all(new_syn1 == syn1)
assert np.all(new_syn2 == syn2)
"""
