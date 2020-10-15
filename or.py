#!/usr/bin/env python3

import numpy as np
import math

#####################################
#   INITIALIZE DATA
#####################################

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

# DEBUG: Verify shapes to do dot product
# correctly.
#print(W.shape)
#print(X.shape)

# sigmoid as a loss function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def forward():
    # Calculate Z
    Z = np.dot(W, X.T) + b

    # define vector of sigmoid function
    Yp = np.vectorize(sigmoid)

    # Evaluate every element of Z in sigmoid function
    Yp = Yp(Z)

    return Yp

def cost(desired_Y, predicted_Y):
    sub = np.subtract(desired_Y.T, predicted_Y)
    square_sub = np.square(sub)

    return np.sum(square_sub) / (2 * desired_Y.size)

def main():
    Yp = forward()
    print("Forward propagation: ", Yp)

    c = cost(Y, Yp)
    print("Cost", c)
    # Now that we have calculated our cost, the objective
    # is to reduce it using backpropagation and gradient
    # descent.


if __name__ == '__main__':
    main()


