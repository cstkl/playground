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
    alpha = 1
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
    #
    # BACKPROPAGATION:

    # Cost node
    dCost_dpY = -np.subtract(Y.T, pY) / Y.size
    dCost_dCost = 1
    upstream_gradient = dCost_dCost
    local_gradient = dCost_dpY
    dCost_dpY = upstream_gradient * local_gradient
    print("dCost/dpY: ", dCost_dpY)

    # Z node
    dpY_dZ = pY * (np.subtract(1, pY))
    upstream_gradient = dCost_dpY
    local_gradient = dpY_dZ
    dCost_dZ = upstream_gradient * local_gradient
    print("dCost/dZ: ", dCost_dZ)

    # Z = WX + b node
    dZ_dW = X
    upstream_gradient = dCost_dZ
    local_gradient = dZ_dW
    dCost_dW = np.dot(upstream_gradient, local_gradient)
    print("dCost/dW: ", dCost_dW)

    dZ_db = 1
    upstream_gradient = dCost_dZ
    local_gradient = dZ_db
    dCost_db = np.sum(upstream_gradient * local_gradient)
    print("dCost/db: ", dCost_db)

    # Now we can update the Weights and bias
    W = np.subtract(W, alpha * dCost_dW)
    b = b - (alpha * dCost_db)
    print("updated W: ", W)
    print("updated b: ", b)

    pY = forward(W, X, b)
    print("FP 2 iteration: ", pY)

    c = cost(Y, pY)
    print("Cost 2 iteration", c)

if __name__ == '__main__':
    main()


