import numpy as np
from .utils import gradient_approximation, gradient_check


class Model:
    def __init__(self, architecture):
        self.architecture = architecture
<<<<<<< HEAD
=======
        # let's initialize layers at first...
        self.init_layers()
        # Workshop #7: implement momentum
        # self.reset_momentum()
>>>>>>> master

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
<<<<<<< HEAD
            A_prev = A
            W = theta["W" + str(layer_idx)]
            b = theta["b" + str(layer_idx)]
            layer_type = self.architecture[layer_idx]['type']

            Z = W @ A_prev + b

            if layer_type == 'linear':
                A = Z
            elif layer_type == 'sigmoid':
                A = 1 / (1 + np.exp(-Z))
            elif layer_type == 'tanh':
                A = np.tanh(Z)
            elif layer_type == 'relu':
                A = np.zeros_like(Z)
                A[Z > 0] = Z[Z > 0]

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
=======
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]

            # Workshop #1: implement back-propagation
            # just apply the formulas with the parameters W_curr and b_curr
            # you need to store a variable called Z_curr for posterity (will be clearer later)
            # and A_curr which is needed in the next iteration of the loop and as a return value

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
        dA_prev = - (np.divide(Y, self.Y_hat) - np.divide(1 - Y, 1 - self.Y_hat))

        # back-propagation algorithm requires that we iterate over layer backwards...
>>>>>>> master
        for layer_idx in range(len(self.architecture)-1, 0, -1):
            dA_curr = dA_prev

            A_prev = self.memory["A" + str(layer_idx-1)]
            Z_curr = self.memory["Z" + str(layer_idx)]

            W_curr = theta["W" + str(layer_idx)]
            b_curr = theta["b" + str(layer_idx)]

            layer_type = self.architecture[layer_idx]['type']

<<<<<<< HEAD
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

    def fit(self, X_train, Y_train, optimizer, no_of_epochs, gradient_check_every, print_every):
        theta = self.init_theta()
        cost = self.get_cost(X_train, Y_train)
        print("Epoch {}, cost: {}".format(0, cost(theta)))
        for epoch_no in range(1, no_of_epochs + 1):
            theta_grad = self.backprop(theta, X_train, Y_train)
            if gradient_check_every and epoch_no % gradient_check_every == 0:
                theta_approx = gradient_approximation(theta, cost)
                gradient_check_results = gradient_check(theta_grad, theta_approx)
                print("Epoch {}, check: {}".format(epoch_no, all(gradient_check_results.values())))

            theta = optimizer.step(theta, theta_grad)

            if print_every and epoch_no % print_every == 0:
                print("Epoch {}, cost: {}".format(epoch_no, cost(theta)))

        return theta
=======
            # Workshop #3: implement back-propagation
            # some suggestions:
            # dA_curr is already computed correctly
            # dZ_curr depends on the activation function aka layer type
            # dW_curr is just like the formula
            # db_curr is just like the formula, but pay extra care on
            # the dimensions of numpy array
            # call dA[l-1] as dA_prev, the assignment at the beginning of the loop
            # will do the rest
            raise Exception("Not implemented")

            self.grads_values["dW" + str(layer_idx)] = dW_curr
            self.grads_values["db" + str(layer_idx)] = db_curr

    def optimization_step(self, learning_rate):
        # Workshop #2: implement vanilla gradient descent step
        # Hint: you need grads_values and params_values...
        raise Exception("Not implemented")
        for layer_idx in range(1, len(self.architecture)):
            pass

    def reset_momentum(self):
        # Workshop #7: implement momentum
        raise Exception("Not implemented")

    def optimization_step_momentum(self, learning_rate, decay_rate=0.9):
        # Workshop #7: implement momentum
        raise Exception("Not implemented")

    def fit(self, X, Y, no_epochs, learning_rate, mini_batch_size=32, print_every=100):
        # WORKSHOP #6
        raise Exception("Not implemented")
>>>>>>> master
