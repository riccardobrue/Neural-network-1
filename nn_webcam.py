import numpy as np
import nn_functions as nnf
import nn_parameters
from PIL import Image
import cv2
import time
import pylab as pl

# ==========================
# WEIGHTS FILENAME PARAMETERS
# ==========================
store_data = False # If true: stores frames for training; if false: runtime recognition based on training

if store_data:
    # filename = 'WithoutPerson.txt'
    filename = 'WithPerson.txt'
    stored_frames = []
    img_counter = 0
else:
    # ==========================
    # RETRIEVE AND SETTING WEIGHTS FROM FILE
    # ==========================
    # filename format: "num_era num_train input x hidden x output"
    filename_num_epochs = nn_parameters.num_epochs  # number of eras in the training phase
    filename_input_num_testing = nn_parameters.input_num_training  # number of images in the training set
    _image_dimension = 64  # 8x8 pixels --> 64 pixels in total --> 64 features
    hidden_units = nn_parameters.hidden_units  # set the number of hidden layers (between 64 and 10)
    _labels_num = 2  # output classes (0,1,2,3,4,5,6,7,8,9)--> 10 labels

    _input_units = _image_dimension  # because the images are 8x8 pixes = 64 pixels
    _output_units = _labels_num  # numbers from 0 to 9

    # setting the correct filename given the parameters
    postfix_filename = '_WebCam_epochs_' + str(filename_num_epochs) + '_num_train_' + str(
        filename_input_num_testing) + '_' + str(_input_units) + 'x' + str(hidden_units) + 'x' + str(
        _output_units) + '.txt'

    input_num_testing = nn_parameters.input_num_testing  # number of images in the testing set

    Theta1_filename = 'Theta1' + postfix_filename
    Theta2_filename = 'Theta2' + postfix_filename
    # Read the array from disk
    Theta1_raw = np.loadtxt(Theta1_filename)
    Theta2_raw = np.loadtxt(Theta2_filename)
    # re-set the weights from file
    Theta1 = np.matrix(Theta1_raw.reshape((hidden_units, (_input_units + 1))))
    Theta2 = np.matrix(Theta2_raw.reshape((_output_units, (hidden_units + 1))))

# ==========================
# GET WEBCAM FRAMES
# ==========================
cam = cv2.VideoCapture(0)
cv2.namedWindow("NN webcam")
start_time = time.time()
while True:
    ret, frame = cam.read()  # frame is a (480,640,3) matrix RGB
    cv2.imshow("NN webcam", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elapsed_time = time.time() - start_time
    if elapsed_time > .1:  # every 1 second
        # ==========================
        # MANIPULATE FRAME
        # ==========================
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert image to a gray scale (480,640) matrix gray
        pil_im = Image.fromarray(gray_frame)  # create a PIL image object from the gray frame array
        frame_img = pil_im.resize((8, 8), Image.ANTIALIAS)  # compress the image into a 8x8 image
        frame = np.asarray(frame_img)  # create an array from the PIL image
        frame_dim = frame.shape[0] * frame.shape[1]  # 480*640 = 307200 if not compresses --> 64 if compressed with PIL
        # convert from matrices into arrays
        frame_arr = np.ravel(frame).reshape((1, frame_dim))
        input_img = (16 / 255) * np.array(frame_arr)

        if store_data:
            # ==========================
            # STORE FRAME
            # ==========================
            if img_counter > 0:  # skip the first black frame
                stored_frames.append(input_img)
            img_counter += 1
        else:
            # ==========================
            # PREDICT FRAME
            # ==========================
            h = nnf.forward_propagate(input_img, Theta1, Theta2, output_only=True)
            # Returns the indices of the maximum values along an axis and creates an array
            y_testing_predicted = np.array(np.argmax(h, axis=1))
            y_predicted_testing_norm = np.squeeze(np.asarray(y_testing_predicted))
            print("Prediction: " + str(y_predicted_testing_norm))
        start_time = time.time()

cam.release()
cv2.destroyAllWindows()

if store_data:
    # ==========================
    # STORING FRAMES INSIDE FILE
    # ==========================
    np_arr = np.array(stored_frames)
    print(np_arr.shape)
    np.savetxt(filename, np.matrix(np_arr))
