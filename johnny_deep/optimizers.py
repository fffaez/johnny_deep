import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, theta, theta_grad):
        for k in theta_grad.keys():
            theta[k] -= self.learning_rate * theta_grad[k]
        return theta


class Momentum:
    def __init__(self, learning_rate, decay_rate=0.1):
        self.velocity = {}
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def step(self, theta, theta_grad):
        if not self.velocity:
            for k in theta_grad.keys():
                self.velocity[k] = np.zeros_like(theta_grad[k])

        for k in theta_grad.keys():
            self.velocity[k] = self.decay_rate * self.velocity[k] + (1-self.decay_rate) * theta_grad[k]
            theta[k] -= self.learning_rate * self.velocity[k]

        return theta
