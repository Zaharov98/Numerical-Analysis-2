"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Fredholm integral equation
    second kind and simpson quadrature formula
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import simpson


def k(t, s):
    return (math.sqrt(s) - 1) / s - t**2


def f(t):
    return 1 - t**2 / (t + 1)


def get_system_matrix(a, x):
    rows = []
    for i in range(0, len(x)):
        row = []
        for j in range(0, len(x)):
            row.append(a[j] * k(x[i], x[j]) if i != j else 1 - a[j] * k(x[i], x[j]))
        rows.append(row)

    return np.array(rows)


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


def main():
    a, b = 2, 5
    h = 0.2

    n = int((b - a) / h)
    m = int(n / 2)

    x = np.arange(a, b, h)
    a = simpson.get_coefficients(h, n)
    matrix = get_system_matrix(a, x)
    solution_row = f(x)

    y = np.linalg.solve(matrix, solution_row)
    display_plot_async(x, y, "Fredholm integral")


if __name__ == '__main__':
    main()
