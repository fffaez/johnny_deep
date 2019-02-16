import numpy as np

from .activations import sigmoid

class Model():
    def __init__(self, architecture: list):
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
            # dimension of previous layer is the input dimension
            layer_previous = self.architecture[layer_idx-1]
            # that's the current layer
            layer = self.architecture[layer_idx]

            # extracting the number of units in layers
            layer_input_size = layer_previous["dimension"]
            layer_output_size = layer["dimension"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            self.params_values['W' + str(layer_idx)] = \
                np.random.randn(layer_output_size, layer_input_size) * 0.1
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
            # that's the current layer
            layer = self.architecture[layer_idx]

            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]

            # calculation of activation for the current layer
            Z_curr = np.dot(W_curr, A_prev) + b_curr

            # selection of activation function
            if layer["type"] is "linear":
                A_curr, Z_curr = Z_curr, Z_curr
            elif layer["type"] is "sigmoid":
                A_curr, Z_curr = sigmoid(Z_curr), Z_curr
            else:
                raise Exception('Non-supported activation function')

            # saving calculated values in the memory
            self.memory["A" + str(layer_idx)] = A_prev
            self.memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector
        return A_curr
