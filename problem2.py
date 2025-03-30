import numpy as np
from matplotlib import pyplot as plt

dataset = [(1, 3), (2, 7), (3, 7), (4, 11), (5, 14), (6, 21), (7, 18), (8, 18), (9, 19), (10, 23)]


def plot_regression_tenth_order(a, b, c, d, e, f, g, h, i, j, k):
    x_values = [point[0] for point in dataset]
    y_values = [point[1] for point in dataset]
    plt.scatter(x_values, y_values, color='red', label='Data Points')

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = (a * x_line**10 + b * x_line**9 + c * x_line**8 + d * x_line**7 + e * x_line**6 + f * x_line**5 +
              g * x_line**4 + h * x_line**3 + i * x_line**2 + j * x_line + k )

    plt.plot(x_line, y_line, color='blue', label=f'Regression Line: 10th Order Polynomial')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Tenth-Order Polynomial Regression')
    plt.savefig("tenth_order.png")


def plot_regression_linear(a, b):
    x_values = [point[0] for point in dataset]
    y_values = [point[1] for point in dataset]
    plt.scatter(x_values, y_values, color='red', label='Data Points')

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color='blue', label=f'Regression Line: y={a:.2f}x+{b:.2f}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Linear Regression')
    plt.savefig("ax_b.png")


def tenth_order(a, b, c, d, e, f, g, h, i, j, k, x, y):
    return (a*(x**10) + b*(x**9) + c*(x**8) + d*(x**7) + e*(x**6) + f*(x**5) +
            g*(x**4) + h*(x**3) + i*(x**2) + j*x + k - y)


def partial_derivative_tenth_regularization(a, b, c, d, e, f, g, h, i, j, k, power, term):
    coefficient = 0.00001
    return partial_derivative_tenth(a, b, c, d, e, f, g, h, i, j, k, power) + 2 * coefficient * term



#a10 b9 c8 d7 e6 f5 g4 h3 i2 j1 k0
def partial_derivative_tenth(a, b, c, d, e, f, g, h, i, j, k, power):
    grad_a_sum = 0
    for instance in dataset:
        grad_a_sum += 2 * (instance[0] ** power) * tenth_order(a, b, c, d, e, f, g, h, i, j, k, instance[0], instance[1])
    return grad_a_sum


def partial_derivative_linear(a, b, power):
    grad_a_sum = 0
    for instance in dataset:
        grad_a_sum += 2 * (instance[0] ** power) * (a * instance[0] + b - instance[1])
    return grad_a_sum


def calculate_parameters_tenth_order_regularization():
    a_0 = 0
    b_0 = 0
    c_0 = 0
    d_0 = 0
    e_0 = 0
    f_0 = 0
    g_0 = 0
    h_0 = 0
    i_0 = 0
    j_0 = 0
    k_0 = 0

    e = 0.000000000000000000001

    while True:
        a_n = a_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 10)
        b_n = b_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 9)
        c_n = c_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 8)
        d_n = d_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 7)
        e_n = e_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 6)
        f_n = f_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 5)
        g_n = g_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 4)
        h_n = h_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 3)
        i_n = i_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 2)
        j_n = j_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 1)
        k_n = k_0 - e * partial_derivative_tenth_regularization(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 0)

        if a_n > 100 or a_n < -100:
            break
        elif ((a_n - a_0) ** 2 + (b_n - b_0) ** 2 + (c_n - c_0) ** 2 + (d_n - d_0) ** 2 + (e_n - e_0) ** 2 +
            (f_n - f_0) ** 2 + (g_n - g_0) ** 2 + (i_n - i_0) ** 2 + (j_n - j_0) ** 2) + (k_n - k_0) ** 2 < e:
            break
        else:
            a_0 = a_n
            b_0 = b_n
            c_0 = c_n
            d_0 = d_n
            e_0 = e_n
            f_0 = f_n
            g_0 = g_n
            h_0 = h_n
            i_0 = i_n
            j_0 = j_n
            k_0 = k_n

    return a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n, i_n, j_n, k_n


def calculate_parameters_tenth_order():
    a_0 = 0
    b_0 = 0
    c_0 = 0
    d_0 = 0
    e_0 = 0
    f_0 = 0
    g_0 = 0
    h_0 = 0
    i_0 = 0
    j_0 = 0
    k_0 = 0

    e = 0.000000000000000000001

    while True:
        a_n = a_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 10)
        b_n = b_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 9)
        c_n = c_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 8)
        d_n = d_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 7)
        e_n = e_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 6)
        f_n = f_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 5)
        g_n = g_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 4)
        h_n = h_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 3)
        i_n = i_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 2)
        j_n = j_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 1)
        k_n = k_0 - e * partial_derivative_tenth(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0, j_0, k_0, 0)

        if a_n > 100 or a_n < -100:
            break
        elif ((a_n - a_0) ** 2 + (b_n - b_0) ** 2 + (c_n - c_0) ** 2 + (d_n - d_0) ** 2 + (e_n - e_0) ** 2 +
            (f_n - f_0) ** 2 + (g_n - g_0) ** 2 + (i_n - i_0) ** 2 + (j_n - j_0) ** 2) + (k_n - k_0) ** 2 < e:
            break
        else:
            a_0 = a_n
            b_0 = b_n
            c_0 = c_n
            d_0 = d_n
            e_0 = e_n
            f_0 = f_n
            g_0 = g_n
            h_0 = h_n
            i_0 = i_n
            j_0 = j_n
            k_0 = k_n

    return a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n, i_n, j_n, k_n

def calculate_parameters_linear():
    a_0 = 0
    b_0 = 0

    print("Initial a =", a_0, "and b =", b_0)

    e = 0.001

    while True:
        a_n = round(a_0 - e * partial_derivative_linear(a_0, b_0, 1), 2)
        b_n = round(b_0 - e * partial_derivative_linear(a_0, b_0, 0), 2)

        if (a_n - a_0)**2 + (b_n - b_0)**2 < e:
            print("a =", a_n, "b =", b_n)
            break
        else:
            a_0 = a_n
            b_0 = b_n

    return a_n, b_n


if __name__ == '__main__':
    a, b = calculate_parameters_linear()
    #plot_regression_linear(a, b)

    a, b, c, d, e, f, g, h, i, j, k = calculate_parameters_tenth_order_regularization()
    plot_regression_tenth_order(a, b, c, d, e, f, g, h, i, j, k)
