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

    print ("*** Predicted Y (Forward propagation) ***")
    print(Yp)

def main():
    forward()

if __name__ == '__main__':
    main()


