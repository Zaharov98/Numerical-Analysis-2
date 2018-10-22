"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Finite difference method
"""

import math
import numpy.linalg
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from boundary_problem import BoundaryProblem


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


def get_boundary_problem_matrix(x, step, problem: BoundaryProblem ):
    matrix = np.zeros(shape=(len(x), len(x)), dtype=np.float)
    result_row = []

    # fill first len(x) - 2 rows
    for i in range(0, len(x) - 2):
        matrix[i][i] = (1 / step**2 - problem.p(x[i]) / step + problem.q(x))
        matrix[i][i + 1] = (problem.p(x[i]) / step - 2 / step**2)
        matrix[i][i + 2] = 1 / step**2
        result_row.append(problem.f(x[i]))

    # fill last two rows
    matrix[-2][0] = problem.a[0] - problem.a[1] / step
    matrix[-2][1] = problem.a[1] / step
    result_row.append(problem.A)

    matrix[-1][-2] = -problem.b[1]
    matrix[-1][-1] = problem.b[0] + problem.b[1] / step
    result_row.append(problem.B)

    return matrix, np.array(result_row)


def main():
    problem = BoundaryProblem(a=[1, 0], b=[2, 3], A=0.5, B=1.2)

    left, right, step = 0.5, 1.5, 0.1
    x = np.arange(left, right + step, step)

    matrix, result_row = get_boundary_problem_matrix(x, step, problem)
    y = np.linalg.solve(matrix, result_row)

    display_plot_async(x, y, 'Finite difference method')


if __name__ == '__main__':
    main()
