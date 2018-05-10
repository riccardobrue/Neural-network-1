import numpy as np
import random
import nn_functions as nnf
from sklearn.datasets import load_digits

# ==========================
# IMPORTING THE DATA
# ==========================

# import the dataset
digits = load_digits()
(data, targets) = load_digits(return_X_y=True)

# ==========================
# WEIGHTS FILENAME PARAMETERS
# ==========================
# filename format: "num_era num_train input x hidden x output"

filename_num_eras = 200000  # number of eras in the training phase
filename_images_num_training_set = 1500  # number of images in the training set
_image_dimension = 64  # 8x8 pixels --> 64 pixels in total --> 64 features
hidden_nodes_n = 20  # set the number of hidden layers (between 64 and 10)
_labels_num = 10  # output classes (0,1,2,3,4,5,6,7,8,9)--> 10 labels

_input_nodes_n = _image_dimension  # because the images are 8x8 pixes = 64 pixels
_output_nodes_n = _labels_num  # numbers from 0 to 9

# setting the correct filename given the parameters
postfix_filename = '_eras_' + str(filename_num_eras) + '_num_train_' + str(
    filename_images_num_training_set) + '_' + str(_input_nodes_n) + 'x' + str(hidden_nodes_n) + 'x' + str(
    _output_nodes_n) + '.txt'

images_num_testing_set = 10  # number of images in the testing set

# ==========================
# SELECTING THE DATA
# ==========================

random_numbers = np.array(random.sample(range(0, len(digits.images)), images_num_testing_set))

testing_indexes = random_numbers[:images_num_testing_set]

# select the images and their targets for the testing sets
chosen_testing_images = digits.images[testing_indexes]
chosen_testing_outputs = targets[testing_indexes]

# convert from matrices into arrays
# testing sets
testing_input = nnf.even_data(chosen_testing_images, num=images_num_testing_set, img_dim=_image_dimension)
testing_outputs = nnf.even_data(chosen_testing_outputs, num=images_num_testing_set, is_input=False)

# ==========================
# RETRIEVE AND SETTING WEIGHTS FROM FILE
# ==========================
syn0_filename = 'syn0' + postfix_filename
syn1_filename = 'syn1' + postfix_filename

# Read the array from disk
syn0_raw = np.loadtxt(syn0_filename)
syn1_raw = np.loadtxt(syn1_filename)

# re-set the weights from file
syn0 = syn0_raw.reshape((_input_nodes_n, hidden_nodes_n))
syn1 = syn1_raw.reshape((hidden_nodes_n, _output_nodes_n))

# ==========================
# TESTING IMAGES
# ==========================
print("Testing unknown images")
(a1, a2, a3) = nnf.compute_prediction_2(testing_input, syn0, syn1)
# print("Output after testing")
a3_testing = nnf.revise_output(a3)
Yout_testing = nnf.postprocess_y(a3_testing)
# print("Original (testing): ")
# print(chosen_testing_outputs)
# print("Calculated (testing): ")
# print(Yout_testing)
accuracy_testing = nnf.get_accuracy(chosen_testing_outputs, Yout_testing)
print("Testing accuracy: " + str(accuracy_testing) + "%")
