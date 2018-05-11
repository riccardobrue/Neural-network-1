import numpy as np
import nn_functions as nnf
import nn_parameters
from PIL import Image, ImageOps
import pylab as pl

# ==========================
# WEIGHTS FILENAME PARAMETERS
# ==========================
# filename format: "num_era num_train input x hidden x output"

filename_num_epochs = nn_parameters.num_epochs  # number of eras in the training phase
filename_input_num_testing = nn_parameters.input_num_training  # number of images in the training set
_image_dimension = 64  # 8x8 pixels --> 64 pixels in total --> 64 features
hidden_units = nn_parameters.hidden_units  # set the number of hidden layers (between 64 and 10)
_labels_num = 10  # output classes (0,1,2,3,4,5,6,7,8,9)--> 10 labels

_input_units = _image_dimension  # because the images are 8x8 pixes = 64 pixels
_output_units = _labels_num  # numbers from 0 to 9

# setting the correct filename given the parameters
postfix_filename = '_epochs_' + str(filename_num_epochs) + '_num_train_' + str(
    filename_input_num_testing) + '_' + str(_input_units) + 'x' + str(hidden_units) + 'x' + str(
    _output_units) + '.txt'

input_num_testing = nn_parameters.input_num_testing  # number of images in the testing set

# ==========================
# LOAD IMAGE
# ==========================
image_name = "img_5.png"

# load image, convert to grayscale, invert colors and put max value to 16 instead of 255
img = ImageOps.invert(Image.open(image_name).convert('L'))
img_dim = img.size[0] * img.size[1]
# convert from matrices into arrays
img_arr_gray = ((np.reshape(np.array(img.getdata()), img.size)) * 16) / 255
input_img = np.ravel(img_arr_gray).reshape((1, img_dim))

# ==========================
# VISUALIZE IMAGE
# ==========================
# pl.gray()
# pl.matshow(img_arr_gray)
# pl.show()

# ==========================
# RETRIEVE AND SETTING WEIGHTS FROM FILE
# ==========================
Theta1_filename = 'Theta1' + postfix_filename
Theta2_filename = 'Theta2' + postfix_filename

# Read the array from disk
Theta1_raw = np.loadtxt(Theta1_filename)
Theta2_raw = np.loadtxt(Theta2_filename)

# re-set the weights from file
Theta1 = np.matrix(Theta1_raw.reshape((hidden_units, (_input_units + 1))))
Theta2 = np.matrix(Theta2_raw.reshape((_output_units, (hidden_units + 1))))

# ==========================
# TESTING PHASE (UNKNOWN IMAGES)
# ==========================
print("Testing phase (...)")
h = nnf.forward_propagate(input_img, Theta1, Theta2, output_only=True)
# Returns the indices of the maximum values along an axis and creates an array
y_testing_predicted = np.array(np.argmax(h, axis=1))
y_predicted_testing_norm = np.squeeze(np.asarray(y_testing_predicted))
print("Prediction: " + str(y_predicted_testing_norm))
