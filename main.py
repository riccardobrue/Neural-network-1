# Neural network - First exercise

import numpy as np


def derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# input data
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# setting a default random each program start
np.random.seed(1)

# building a 3-4-1 neural network
# synapses matrices
syn0 = 2 * np.random.random((3, 4)) - 1  # 3x4 matrix
syn1 = 2 * np.random.random((4, 1)) - 1  # 4x1 matrix

# training step
for j in range(60000):
    a1 = X
    a2 = sigmoid(np.dot(a1, syn0))
    a3 = sigmoid(np.dot(a2, syn1))

    a3_error = y - a3
    a3_delta = a3_error * sigmoid(derivative(a3))

    a2_error = a3_delta.dot(syn1.T)  # transposed matrix
    a2_delta = a2_error * sigmoid(derivative(a2))

    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(a3_error))))

    # update weights
    syn1 += a2.T.dot(a3_delta)
    syn0 += a1.T.dot(a2_delta)

print("Output after training")
print(a3)

a3[a3 >= 0.5] = 1
a3[a3 < 0.5] = 0

print(a3)
