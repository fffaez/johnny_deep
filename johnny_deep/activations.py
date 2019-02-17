import numpy as np


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu(Z):
    # WORKSHOP #5: code the ReLU activation function
    raise Exception("Not implemented")


def relu_backward(dA, Z):
    # WORKSHOP #5: code the ReLU activation function
    raise Exception("Not implemented")
