#!/usr/bin/env python3

import numpy as np
import math

# The lost function consist of a equation that tell us how far
# our NN predicted output "pY" is from our desired output "Y".
# There are different functions that can do this job, however
# in this case I am using sigmoid.
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# In order to use batch gradient descent, we need to change a
# little our loss function. The idea is to adapt it for all
# samples not only for one. This adjusted function is called
# "cost function". The cost function is helps to minimize the
# vertical distance(Squared Error Loss) between multiple data
# points with respect the predictor line.
def cost(Y, pY):
    sub = np.subtract(Y.T, pY)
    square_sub = np.square(sub)

    return np.sum(square_sub) / (2 * Y.size)


def forward(W, X, b):
    # Calculate Z
    Z = np.dot(W, X.T) + b

    # define vector of sigmoid function
    pY = np.vectorize(sigmoid)

    # Evaluate every element of Z in sigmoid function
    pY = pY(Z)

    return pY


def main():
    # Random initial values for wights
    W = np.array([0.1, 0.6])
    W = np.reshape(W, (1, 2))

    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])

    b = 0
    Y = np.array([0, 1, 1, 1])
    Y = np.reshape(Y, (4, 1))

    pY = forward(W, X, b)
    print("Forward propagation: ", pY)

    c = cost(Y, pY)
    print("Cost", c)
    # Now that we have calculated our cost, the objective
    # is to reduce it using backpropagation and gradient
    # descent.

    # BACKPROPAGATION
    # PREDICTION NODE
    #
    # Evaluate the derivative of cost function respect to
    # predict_Y
    local_gradient_pn = -np.subtract(Y.T, pY) / Y.size
    print("local gradient predict node: ", local_gradient_pn)

    # This is because the derivative of cost function by
    # itself is 1
    upstream_gradient_pn = 1
    local_gradient_pn = upstream_gradient_pn * local_gradient_pn
    print("upstream * local gradient: ", local_gradient_pn)

    # Z NODE
    # The local gradient from last node is our global
    upstream_gradient_zn = local_gradient_pn
    local_gradient_zn = pY * (np.subtract(1, pY))
    print("local gradient Z node: ", local_gradient_zn)

    local_gradient_zn = upstream_gradient_zn * local_gradient_zn
    print("upstream * local gradient z: ", local_gradient_zn)

    # Now we can calculate the gradient with respect to weights
    # and bias. Z = WX + b
    upstream_gradient_wn = local_gradient_zn
    local_gradient_wn = X
    local_gradient_wn = np.dot(upstream_gradient_wn, local_gradient_wn)
    print("local gradient W: ", local_gradient_wn)

    # gradiend bias
    upstream_gradient_bn = local_gradient_zn
    local_gradient_bn = 1
    local_gradient_bn = upstream_gradient_bn * local_gradient_bn
    #print("local gradient b: ", local_gradient_bn)
    local_gradient_bn = np.sum(local_gradient_bn)
    print("local gradient b: ", local_gradient_bn)

if __name__ == '__main__':
    main()


