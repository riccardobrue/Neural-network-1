import numpy as np


def derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # this gives an overflow --> it is not important
    # return .5 * (1 + np.tanh(.5 * x))  # same as previous --> solves the overflow problem


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


# feed forward function
def compute_prediction_2(X, Theta1, Theta2):
    a1 = X
    # a2 = sigmoid(np.dot(a1, syn0))  # np.dot --> product between two arrays
    # a3 = sigmoid(np.dot(a2, syn1))

    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    return a1, a2, a3


# feed forward function
def compute_prediction_3(X, syn0, syn1, syn2):
    a1 = X
    a2 = sigmoid(np.dot(a1, syn0))  # np.dot --> product between two arrays
    a3 = sigmoid(np.dot(a2, syn1))
    a4 = sigmoid(np.dot(a3, syn2))
    return a1, a2, a3, a4


def revise_output(data):
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    return data


def even_data(data, num=1, img_dim=1, is_input=True):
    if is_input:
        return np.ravel(data).reshape((num, img_dim))
    else:
        return data.reshape((num, 1))
