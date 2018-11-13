"""
    Numeric analysis methods
    Laboratory work number 6
    Variant 6, Zaharov Igor

    Volterra integral equation
    by trapezium quadrature formula
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def q(t, s):
    return t + 2 * math.log(1 + s)


def f(t):
    return 3 * t - math.sqrt(t) - 1


def _display_plot(x_nodes, y_nodes, plot_title):
    """
     Display the plot of interpolated points using pyplot
    :param x_nodes: range of arguments of interpolated function
    :param y_nodes: range of interpolated values
    :param plot_title: title of the plot
    """
    plt.plot(x_nodes, y_nodes, 'o', label='Calculated', linewidth=0.1)
    plt.title(plot_title)

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_nodes, y_nodes, plot_title):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot, args=(x_nodes, y_nodes, plot_title))
    process.start()


def get_quadrature_weights(h, n):
    weights = [h / 2, ] + [h, ] * (n - 1)
    weights.append(h / 2)

    return np.array(weights)


def get_nodes(x, w, h):
    nodes = []
    for i in range(0, len(x)):
        node = 1 / (1 - (h / 2) * q(x[i], x[i])) * (f(x[i]) + h / 2 * q(x[i], x[1]) * f(x[1]) +
                                                    h * sum([q(x[i], x[j]) * f(x[j]) for j in range(0, i)]))
        nodes.append(node)

    return nodes


def main():
    a, b = 2, 4
    h = 0.1

    n = int((b - a) / h)

    x = np.arange(a, b, h)
    w = get_quadrature_weights(h, n)
    y = get_nodes(x, w, h)

    display_plot_async(x, y, 'Volterra integral equation')


if __name__ == '__main__':
    main()
