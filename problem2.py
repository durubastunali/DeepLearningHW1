import numpy as np
from matplotlib import pyplot as plt

dataset = [(1, 3), (2, 7), (3, 7), (4, 11), (5, 14), (6, 21), (7, 18), (8, 18), (9, 19), (10, 23)]


def plot_regression_tenth_order(params):
    x_values = [point[0] for point in dataset]
    y_values = [point[1] for point in dataset]
    plt.scatter(x_values, y_values, color='red', label='Data Points')

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = (params[0] * x_line**10 + params[1] * x_line**9 + params[2] * x_line**8 + params[3] * x_line**7 +
              params[4] * x_line**6 +  params[5] * x_line**5 + params[6] * x_line**4 + params[7] * x_line**3 +
              params[8] * x_line**2 +  params[9] * x_line + params[10])

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


def partial_derivative_a_linear(a_n, b_n):
    grad_a_sum = 0
    for instance in dataset:
        grad_a_sum += 2 * a_n * (a_n * instance[0] + b_n - instance[1])
    return grad_a_sum


def partial_derivative_b_linear(a_n, b_n):
    grad_b_sum = 0
    for instance in dataset:
        grad_b_sum += 2 * (a_n * instance[0] + b_n - instance[1])
    return grad_b_sum


def calculate_parameters_tenth_order():
    a_0 = np.random.uniform(-5, 5)
    b_0 = np.random.uniform(-5, 5)
    c_0 = np.random.uniform(-5, 5)
    d_0 = np.random.uniform(-5, 5)
    e_0 = np.random.uniform(-5, 5)
    f_0 = np.random.uniform(-5, 5)
    g_0 = np.random.uniform(-5, 5)
    h_0 = np.random.uniform(-5, 5)
    i_0 = np.random.uniform(-5, 5)
    j_0 = np.random.uniform(-5, 5)
    k_0 = np.random.uniform(-5, 5)

    e = 0.0001
    params = []

    while True:
        grads = []
        power = 10

        while power >= 0:
            grad_sum = 0
            for instance in dataset:
                grad_sum += (a_0*(instance[0]**10) + b_0*(instance[0]**9) + c_0*(instance[0]**8) + d_0*(instance[0]**7) +
                             e_0*(instance[0]**6) + f_0*(instance[0]**5) + g_0*(instance[0]**4) + h_0*(instance[0]**3) +
                             i_0*(instance[0])**2 + j_0*instance[0] + k_0 - instance[1]) * 2 * (instance[0] ** power)
            grads.append(grad_sum)
            power -= 1

        a_n = a_0 - e * grads[0]
        b_n = b_0 - e * grads[1]
        c_n = c_0 - e * grads[2]
        d_n = d_0 - e * grads[3]
        e_n = e_0 - e * grads[4]
        f_n = f_0 - e * grads[5]
        g_n = g_0 - e * grads[6]
        h_n = h_0 - e * grads[7]
        i_n = i_0 - e * grads[8]
        j_n = j_0 - e * grads[9]
        k_n = k_0 - e * grads[10]

        if ((a_n - a_0)**2 + (b_n - b_0)**2 + (c_n - c_0)**2 + (d_n - d_0)**2 + (e_n - e_0)**2 + (f_n - f_0)**2 +
            (g_n - g_0)**2 + (h_n - h_0)**2 + (i_n - i_0)**2 + (j_n - j_0)**2 + (k_n - k_0)**2) < e:
            params = [a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n, i_n, j_n, k_n]
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

    return params


def calculate_parameters_linear():
    a_0 = np.random.uniform(-1, 1)
    b_0 = np.random.uniform(-1, 1)

    e = 0.0001

    while True:
        a_n = a_0 - e * partial_derivative_a_linear(a_0, b_0)
        b_n = b_0 - e * partial_derivative_b_linear(a_0, b_0)

        if (a_n - a_0)**2 + (b_n - b_0)**2 < e:
            print("a =", a_n, "b =", b_n)
            break
        else:
            a_0 = a_n
            b_0 = b_n

    return a_n, b_n


if __name__ == '__main__':
    a, b = calculate_parameters_linear()
    plot_regression_linear(a, b)

    params = calculate_parameters_tenth_order()
    plot_regression_tenth_order(params)
