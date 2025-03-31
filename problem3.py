import numpy as np
from matplotlib import pyplot as plt

dataset = [(10, 0), (20, 0), (15, 0), (40,0), (50, 1), (60, 0), (60, 1), (70,1), (80,0), (90, 1), (95, 1), (100,1), (100, 1)]


def sigmoid(i, p, q):
    return 1 / (1 + np.exp(-(p*i + q)))


def gradient_p(p, q):
    sum = 0
    for instance in dataset:
        x, y = instance[0], instance[1]
        sum += (sigmoid(x, p, q) - y) * x
    return sum


def gradient_q(p, q):
    sum = 0
    for instance in dataset:
        x, y = instance[0], instance[1]
        sum += (sigmoid(x, p, q) - y)
    return sum


def logistic_regression():
    p = np.random.uniform(-5, 5)
    q = np.random.uniform(-5, 5)

    e = 0.01

    while True:
        p_new = p - e * gradient_p(p, q)
        q_new = q - e * gradient_q(p, q)

        if (p - p_new)**2 + (q - q_new)**2 < e:
            print("p:", p_new, "q:", q_new)
            break
        else:
            p = p_new
            q = q_new
    return p, q


def plot_sigmoid(p, q):
    x_values = [point[0] for point in dataset]
    y_values = [point[1] for point in dataset]
    plt.scatter(x_values, y_values, color='red', label='Data Points')

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = 1 / (1 + np.exp(-(p*x_line + q)))
    plt.plot(x_line, y_line, color='blue', label=f'Regression Line: p={p:.2f}, q={q:.2f}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Logistic Regression - Sigmoid')
    plt.savefig("logistic_regression.png")


if __name__ == '__main__':
    p, q = logistic_regression()
    plot_sigmoid(p, q)
