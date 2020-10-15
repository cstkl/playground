#!/usr/bin/env python3

import numpy as np
import math


# sigmoid as a loss function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

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

# Verify shapes to do dot product correctly
print(W.shape)
print(X.shape)

Z = np.dot(W, X.T) + b

# define vector of sigmoid function
Yp = np.vectorize(sigmoid)

# Evaluate every element of Z in sigmoid function
Yp = Yp(Z)

print ("*** Predicted Y (Forward propagation) ***")
print(Yp)


