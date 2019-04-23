import numpy as np
from .utils import gradient_approximation, gradient_check


class Model:
    def __init__(self, architecture):
        self.architecture = architecture

    def init_theta(self, seed=42, verbose=True):
        np.random.seed(seed)
        theta = {}

        for layer_idx in range(1, len(self.architecture)):
            layer_input_size = self.architecture[layer_idx-1]["dimension"]
            layer_output_size = self.architecture[layer_idx]["dimension"]
            layer_type = self.architecture[layer_idx]['type']

            theta['W' + str(layer_idx)] = \
                np.random.randn(layer_output_size, layer_input_size) * 0.1
            theta['b' + str(layer_idx)] = \
                np.zeros((layer_output_size, 1))

            if verbose:
                print("Layer {}: {} dimension {}".format(layer_idx, layer_type, self.architecture[layer_idx]["dimension"]))
                print("W shape: {}".format(theta['W' + str(layer_idx)].shape))
                print("b shape: {}".format(theta['b' + str(layer_idx)].shape))

        return theta

    def forward(self, theta, X):
        self.memory = {}
        A = X
        self.memory["A" + str(0)] = X

        for layer_idx in range(1, len(self.architecture)):
            A_prev = A
            W = theta["W" + str(layer_idx)]
            b = theta["b" + str(layer_idx)]
            layer_type = self.architecture[layer_idx]['type']

            # Workshop #1: implement forward pass
            Z = W @ A_prev + b
            if layer_type == 'linear':
                A = Z
            elif layer_type == 'sigmoid':
                A = 1 / (1 + np.exp(-Z))
            elif layer_type == 'tanh':
                A = np.tanh(Z)
            elif layer_type == 'relu':
                A = np.copy(Z)
                A[Z<0] = 0
            # Workshop #2: END

            self.memory["A" + str(layer_idx)] = A
            self.memory["Z" + str(layer_idx)] = Z

        return A


    def backprop(self, theta, X, Y):
        m = X.shape[1]
        Y_hat = self.forward(theta, X)

        grads_values = {}
        for k, param in theta.items():
            grads_values[k] = np.zeros_like(param)

        dA_prev = - ((Y / Y_hat) - ((1 - Y) / (1 - Y_hat)))

        for layer_idx in range(len(self.architecture)-1, 0, -1):
            dA_curr = dA_prev

            A_prev = self.memory["A" + str(layer_idx-1)]
            Z_curr = self.memory["Z" + str(layer_idx)]

            W_curr = theta["W" + str(layer_idx)]
            b_curr = theta["b" + str(layer_idx)]

            layer_type = self.architecture[layer_idx]['type']

            # Workshop #6: implement back-propagation
            # remember that we're looping layers backwards...

            if layer_type == 'linear':
                dZ_curr = dA_curr
            elif layer_type == 'sigmoid':
                dZ_curr = dA_curr * np.exp(-Z_curr) / (np.exp(-Z_curr) + 1)**2
            elif layer_type == 'tanh':
                dZ_curr = dA_curr * (1 - np.tanh(Z_curr)**2)
            elif layer_type == 'relu':
                dZ_curr = np.array(dA_curr, copy = True)
                dZ_curr[Z_curr < 0] = 0

            # derivative of the matrix W
            dW_curr = 1/m * np.dot(dZ_curr, A_prev.T)
            # derivative of the vector b
            db_curr = 1/m * np.sum(dZ_curr, axis=1, keepdims=True)
            # derivative of the matrix A_prev
            dA_prev = np.dot(W_curr.T, dZ_curr)

            grads_values["W" + str(layer_idx)] = dW_curr
            grads_values["b" + str(layer_idx)] = db_curr

        return grads_values


    def get_cost(self, X, Y):
        m = X.shape[1]

        def cost(theta):
            Y_hat = self.forward(theta, X)
            return - 1/m * np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat))

        return cost

    def fit(self, X_train, Y_train, optimizer, no_of_epochs, print_every, gradient_check_every=None):
        theta = self.init_theta()
        cost = self.get_cost(X_train, Y_train)
        print("Epoch {}, cost: {}".format(0, cost(theta)))
        for epoch_no in range(1, no_of_epochs + 1):

            theta_grad = self.backprop(theta, X_train, Y_train)
            theta = optimizer.step(theta, theta_grad)

            if print_every and epoch_no % print_every == 0:
                print("Epoch {}, cost: {}".format(epoch_no, cost(theta)))

            if gradient_check_every and epoch_no % gradient_check_every == 0:
                theta_grad_approx = gradient_approximation(theta, cost)
                print(gradient_check(theta_grad, theta_grad_approx))

        return theta
