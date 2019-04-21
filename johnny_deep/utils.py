import numpy as np

EPSILON = 1E-4


def plot_with_countours(plt, X_test, Y_pred, m, theta):
    Y_pred = np.squeeze(Y_pred)
    fig, ax = plt.subplots()
    ax.scatter(X_test[0, :], X_test[1, :], marker='o', c=Y_pred, s=25, edgecolor='k', zorder=1)

    delta = 0.025
    X1, X2 = np.meshgrid(
        np.arange(min(X_test[0, :]), max(X_test[0, :]), delta),
        np.arange(min(X_test[1, :]), max(X_test[1, :]), delta)
    )

    X_grid = np.append(
            X1.reshape(-1, 1),
            X2.reshape(-1, 1),
            axis=1).T

    Z = m.forward(theta, X_grid)

    ax.contour(X1, X2, Z.reshape(X1.shape), zorder=2)
    plt.show()



def gradient_approximation(theta, cost):
    theta_grad, theta_plus, theta_minus = {}, {}, {}
    for k, param in theta.items():
        theta_grad[k] = np.zeros_like(param)
        theta_plus[k] = np.copy(param)
        theta_minus[k] = np.copy(param)

    for k, param in theta.items():
        for idx, _ in np.ndenumerate(param):
            theta_plus[k] = np.copy(param)
            theta_plus[k][idx] += EPSILON

            theta_minus[k] = np.copy(param)
            theta_minus[k][idx] -= EPSILON

            theta_grad[k][idx] = (cost(theta_plus) - cost(theta_minus)) / (2 * EPSILON)

            theta_plus[k] = param
            theta_minus[k] = param

    return theta_grad


def gradient_check(theta_a, theta_b, threshold):
    theta_check = {}
    for k in theta_a:
        theta_check[k] = theta_a[k] - theta_b[k]
        theta_check[k] = np.all(theta_check[k] < threshold)
    return theta_check


def fit(model, X_train, Y_train, no_of_epochs, gradient_check_every, print_every):
    theta = model.init_theta()
    cost = model.get_cost(X_train, Y_train)
    print("Epoch {}, cost: {}".format(0, cost(theta)))
    for epoch_no in range(1, no_of_epochs + 1):
        theta_grad = model.backprop(theta, X_train, Y_train)
        if epoch_no % gradient_check_every == 0:
            theta_approx = gradient_approximation(theta, cost)
            gradient_check_results = gradient_check(theta_grad, theta_approx, EPSILON*10)
            print("Epoch {}, check: {}".format(epoch_no, all(gradient_check_results.values())))
        for k in theta.keys():
            theta[k] -= 0.3 * theta_grad[k]
        if epoch_no % print_every == 0:
            print("Epoch {}, cost: {}".format(epoch_no, cost(theta)))

    return theta
