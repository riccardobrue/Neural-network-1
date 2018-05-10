import numpy as np
from sklearn.preprocessing import OneHotEncoder


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # this gives an overflow --> it is not important
    # return .5 * (1 + np.tanh(.5 * x))  # same as previous --> solves the overflow problem


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# one-hot encode our labels
def onehot_y(y):
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(y)


def get_accuracy(y1, y2):
    size = len(y1)
    count = 0
    for x in range(0, size):
        if y1[x] == y2[x]:
            count += 1
    return (count * 100) / size


def even_data(data, num=1, img_dim=1, is_input=True):
    if is_input:
        return np.ravel(data).reshape((num, img_dim))
    else:
        return data.reshape((num, 1))


# feed forward with bias
def forward_propagate(X, Theta1, Theta2, output_only=False):
    m = X.shape[0]  # length of the input array
    # Insert values along the given axis (1->row) before the given indices (0)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * Theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * Theta2.T
    h = sigmoid(z3)
    if output_only:
        return h
    else:
        return a1, z2, a2, z3, h


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate, only_cost=True):
    m = X.shape[0]  # length of the input array
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    Theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    Theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, Theta1, Theta2)

    # compute the cost
    """
    J = (1 / m) * sum(sum((-Y. * log(h)) - ((1 - Y). * log(1 - h)), 2))
    """
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    # add the regularization to penalize irrelevant attributes and shrink parameters
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2)))
    if only_cost:
        return J
    else:
        return J, Theta1, Theta2, a1, a2, h, z2, z3


# train the network
def back_propagate(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    J, Theta1, Theta2, a1, a2, h, z2, z3 = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate,
                                                only_cost=False)

    delta1 = np.zeros(Theta1.shape)  # (hidden_size x input_size+1)
    delta2 = np.zeros(Theta2.shape)  # (num_labels x hidden_size+1)

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1 x input_size+1)
        z2t = z2[t, :]  # (1 x hidden_size)
        a2t = a2[t, :]  # (1 x hidden_size+1)
        ht = h[t, :]  # (1 x num_labels)
        yt = y[t, :]  # (1 x num_labels)

        d3t = ht - yt  # (1 x num_labels)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1 x hidden_size+1)
        d2t = np.multiply((Theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1 x hidden_size+1)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (Theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (Theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad
