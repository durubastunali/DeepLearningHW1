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


def gradient(x_0, y_0):
    print(f"Initial: X={x_0}, Y={y_0}")

    grad = 1  # Descent = 0, Ascent = 1
    e = 0.1

    path = [(x_0, y_0)]

    while True:
        x_new, y_new = calculate_next_xy(x_0, y_0, e, grad)
        stop_condition = (f(x_new, y_new) - f(x_0, y_0))**2

        if stop_condition < 0.001:
            print(f"Final: X={x_new}, Y={y_new}")
            break
        else:
            x_0, y_0 = x_new, y_new
            path.append((x_new, y_new))

    return path


def plot_isohypse():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.exp(-1*(X**2 + Y**2)) + 3*np.exp(-2**((X-2)**2 + (Y-3)**2)) - 4*np.exp(-2*((X+4)**2 + (Y+3)**2))

    plt.figure(figsize=(8, 6))
    contours = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gradient Ascent")


def plot_gradient_path(path):
    x_path, y_path = zip(*path)
    plt.plot(x_path, y_path, linestyle="-", color="r", marker="o", markersize=3, label="Gradient Path")

    for i in range(len(x_path) - 1):
        plt.annotate(
            "",
            xy=(x_path[i + 1], y_path[i + 1]),
            xytext=(x_path[i], y_path[i]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
        )
    plt.legend()


if __name__ == '__main__':
    start_points = []
    for i in range(5):
        start_points.append((np.random.uniform(-5, 5), np.random.uniform(-5)))
    plot_isohypse()
    for x_0, y_0 in start_points:
        path = gradient(x_0, y_0)
        plot_gradient_path(path)
    plt.savefig("isohypse_plot.png")
