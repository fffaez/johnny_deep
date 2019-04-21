import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, theta, theta_grad):
        for k in theta_grad.keys():
            # Workshop #3: implement the vanilla gradient descent step
            # store the new value of theta in theta[k]...
            raise Exception('Not implemented yet!')
            # Workshop #3: END
        # don't forget to return theta!
        return theta
