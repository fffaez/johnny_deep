import numpy as np

EPSILON = 1E-4


def plot_with_countours(plt, X_test, Y_pred, m, theta):
    Y_pred = np.squeeze(Y_pred)
    _, ax = plt.subplots()
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
            theta_minus[k] = np.copy(param)

            # Workshop #4:
            # remember that inside this loop
            # k is the parameter matrix name i.e. 'W_1'
            # idx is the index (tuple) inside such an array (thanks to ndenumerate)
            # So you have to:
            # add a tiny EPSILON to theta_plus
            # substract a tiny EPSILON to theta_minus
            # now you compute theta_grad[k][idx] as an approximation of the gradient
            # Workshop #4: END

            # let's not forget to copy the original parameter back into
            # theta_plus and theta_minus, so we're ready for another loop...
            theta_plus[k] = param
            theta_minus[k] = param

    return theta_grad


def gradient_check(theta_a, theta_b, threshold=EPSILON*10):
    theta_check = {}
    for k in theta_a:
        theta_check[k] = theta_a[k] - theta_b[k]
        theta_check[k] = np.all(theta_check[k] < threshold)
    return theta_check
