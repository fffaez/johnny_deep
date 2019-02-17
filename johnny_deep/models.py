import numpy as np

from .activations import sigmoid, sigmoid_backward
from .activations import relu, relu_backward

class Model():
    def __init__(self, architecture):
        if len(architecture) < 1 and architecture[0]['type'] != 'input':
            raise Exception("Model architecture must be deeper than one layer and first layer type must be input")
        self.architecture = architecture
        # let's initialize layers at first...
        self.init_layers()


    def init_layers(self, seed=42):
        # random seed initiation
        np.random.seed(seed)
        # number of layers in our neural network, input layer doesn't count
        number_of_layers = len(self.architecture) - 1
        # parameters storage initiation
        self.params_values = {}

        # iteration over network layers
        for layer_idx in range(1, len(self.architecture)):
            # extracting the number of units in layers
            # input size from the previous layer:
            layer_input_size = self.architecture[layer_idx-1]["dimension"]
            # output size from the current layer:
            layer_output_size = self.architecture[layer_idx]["dimension"]

            # initiating the values of the W matrix
            # randomness is important here: otherwise all neurons will learn in the same way
            # try to tweak the random factor, make it a parameter or google for some other heuristics
            # as described here: https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
            self.params_values['W' + str(layer_idx)] = \
                np.random.randn(layer_output_size, layer_input_size) * 0.1
            # initiating the values of b
            # this can be either all zero or random
            self.params_values['b' + str(layer_idx)] = \
                np.zeros((layer_output_size, 1))

    def model_info(self):
        for layer_idx in range(1, len(self.architecture)):
            layer = self.architecture[layer_idx]
            print("Layer {}: {} with dimension {}".format(layer_idx, layer["type"], layer["dimension"]))
            print("W shape: {}".format(self.params_values['W' + str(layer_idx)].shape))
            print("b shape: {}".format(self.params_values['b' + str(layer_idx)].shape))

    def forward(self, X):
        # creating a temporary memory to store the information needed for a backward step
        self.memory = {}
        # X vector is the activation for layer 0 
        A_curr = X

        # iteration over network layers
        for layer_idx in range(1, len(self.architecture)):
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]

            # calculation of activation for the current layer
            Z_curr = np.dot(W_curr, A_prev) + b_curr

            # selection of activation function
            layer_type = self.architecture[layer_idx]["type"]
            if layer_type is "linear":
                A_curr, Z_curr = Z_curr, Z_curr
            elif layer_type is "sigmoid":
                A_curr, Z_curr = sigmoid(Z_curr), Z_curr
            elif layer_type is "relu":
                A_curr, Z_curr = relu(Z_curr), Z_curr
            else:
                raise Exception('Non-supported activation function')

            # saving calculated values in the memory
            self.memory["A" + str(layer_idx-1)] = A_prev
            self.memory["Z" + str(layer_idx)] = Z_curr

        # saving current prediction vector as Y_hat
        # for future back_propagation but also it's
        # the function return value when used for inference
        self.Y_hat = A_curr

        # return Y_hat
        return self.Y_hat

    def back_propagation(self, Y):
        self.grads_values = {}

        # Y_hat has the shape of (ouput_dim, no_examples)
        # because we do binary classification only Y might have the
        # shape of (no_sample), so a reshape of Y is needed here
        Y = Y.reshape(self.Y_hat.shape)

        # number of examples
        m = Y.shape[1]

        # initiation of gradient descent algorithm
        # hardcoded derivative of log_loss wrt Y_hat
        # which is the input of backpropagation algorithm
        dA_prev = - (np.divide(Y, self.Y_hat) - np.divide(1 - Y, 1 - self.Y_hat));

        # back-propagation algorithm requires that we iterate over layer backwards...
        for layer_idx in range(len(self.architecture)-1, 0, -1):
            dA_curr = dA_prev

            # let's grab value of activations and Z of the previous layer
            # that we stored while the forward step...
            A_prev = self.memory["A" + str(layer_idx-1)]
            Z_curr = self.memory["Z" + str(layer_idx)]

            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]

            # number of examples
            m = A_prev.shape[1]

            # selection of activation function
            layer_type = self.architecture[layer_idx]["type"]
            if layer_type is "linear":
                dZ_curr = dA_curr
            elif layer_type is "sigmoid":
                dZ_curr = sigmoid_backward(dA_curr, Z_curr)
            elif layer_type is "relu":
                dZ_curr = relu_backward(dA_curr, Z_curr)
            else:
                raise Exception('Non-supported activation function')

            # derivative of the matrix W
            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            # derivative of the vector b
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            # derivative of the matrix A_prev
            dA_prev = np.dot(W_curr.T, dZ_curr)


            self.grads_values["dW" + str(layer_idx)] = dW_curr
            self.grads_values["db" + str(layer_idx)] = db_curr

    def optimization_step(self, learning_rate):
        # that's a implementation of vanilla gradient descent
        # here's the place where we should consider implementing other
        # state of the art heuristics like momentum, RMSProp and Adam
        # https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/
        for layer_idx in range(1, len(self.architecture)):
            self.params_values["W" + str(layer_idx)] -= learning_rate * self.grads_values["dW" + str(layer_idx)]
            self.params_values["b" + str(layer_idx)] -= learning_rate * self.grads_values["db" + str(layer_idx)]
