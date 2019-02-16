import johnny_deep
from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=1000, noise=0.05, random_state=100)
X = X.transpose()

nn_architecture = [
    {"dimension": 2, "type": "input"},
    {"dimension": 4, "type": "sigmoid"},
    {"dimension": 6, "type": "sigmoid"},
    {"dimension": 6, "type": "sigmoid"},
    {"dimension": 4, "type": "sigmoid"},
    {"dimension": 1, "type": "sigmoid"},
]

m = johnny_deep.models.Model(nn_architecture)

for ix in range(10000):
    Y_hat = m.forward(X)
    cost = johnny_deep.utils.get_cost_value(Y_hat, Y)
    print("Epoch {} - cost {}".format(ix+1, cost))
    m.back_propagation(Y)
    m.optimization_step(0.1)
