import numpy as np
import matplotlib.pyplot as plt


def partial_derivative_x(x_n, y_n):
    term1 = -2 * x_n * np.exp(-x_n ** 2 - y_n ** 2)
    term2 = -12 * (x_n - 2) * np.exp(-2 * (x_n ** 2 - 4 * x_n + y_n ** 2 - 6 * y_n + 13))
    term3 = 16 * (x_n + 4) * np.exp(-2 * (x_n ** 2 + 8 * x_n + y_n ** 2 + 6 * y_n + 25))
    return term1 + term2 + term3


def partial_derivative_y(x_n, y_n):
    term1 = -2 * y_n * np.exp(-x_n ** 2 - y_n ** 2)
    term2 = -12 * (y_n - 3) * np.exp(-2 * (x_n ** 2 - 4 * x_n + y_n ** 2 - 6 * y_n + 13))
    term3 = 16 * (y_n + 3) * np.exp(-2 * (x_n ** 2 + 8 * x_n + y_n ** 2 + 6 * y_n + 25))
    return term1 + term2 + term3


def calculate_next_xy(x_n, y_n, e, grad):
    if grad == 0:  # gradient descent
        x_new = x_n - e * partial_derivative_x(x_n, y_n)
        y_new = y_n - e * partial_derivative_y(x_n, y_n)
    elif grad == 1:  # gradient ascent
        x_new = x_n + e * partial_derivative_x(x_n, y_n)
        y_new = y_n + e * partial_derivative_y(x_n, y_n)
    return x_new, y_new


def f(X, Y):
    return np.exp(-1 * (X ** 2 + Y ** 2)) + 3 * np.exp(-2 ** ((X - 2) ** 2 + (Y - 3) ** 2)) - 4 * np.exp(
        -2 * ((X + 4) ** 2 + (Y + 3) ** 2))


def gradient():
    x_0 = np.random.uniform(-5, 5)
    y_0 = np.random.uniform(-5, 5)

    print("inital: X={}, Y={}".format(x_0, y_0))

    grad = 0  # descent = 0, ascent = 1
    e = 0.01

    # Path to store coordinates of the ascent path
    path = [(x_0, y_0)]

    while True:
        x_new, y_new = calculate_next_xy(x_0, y_0, e, grad)
        stop_condition = (f(x_new, y_new) - f(x_0, y_0))**2

        print("stop cond=", stop_condition)

        # Store the new coordinates in the path
        path.append((x_new, y_new))

        if stop_condition < 0.0001:
            print("x =", x_new, "y =", y_new)
            break
        else:
            x_0 = x_new
            y_0 = y_new

    return path


def plot_isohypse():
    x = np.linspace(-5, 5, 100)  # X range
    y = np.linspace(-5, 5, 100)  # Y range
    X, Y = np.meshgrid(x, y)

    Z = np.exp(-1*(X**2 + Y**2)) + 3*np.exp(-2**((X-2)**2 + (Y-3)**2)) - 4*np.exp(-2*((X+4)**2 + (Y+3)**2))

    plt.figure(figsize=(8, 6))
    contours = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)  # Label the contours

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Isohypse Plot")


def plot_gradient_path(path):
    x_path, y_path = zip(*path)  # Unzip the path into x and y coordinates

    plt.plot(x_path, y_path, marker='o', color='r', markersize=5, label="Gradient Ascent Path")
    plt.legend()


if __name__ == '__main__':
    path = gradient()
    plot_isohypse()
    plot_gradient_path(path)
    plt.savefig("isohypse_plot.png")
