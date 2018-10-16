"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Classical Runge-Kutta method of the n-th order
"""

import numpy as np
import tableprint as tp
import matplotlib.pyplot as plt
from multiprocessing import Process


def _display_plot(x_nodes_list, y_nodes_list):
    """
     Display the plot of interpolated points using pyplot
    :param x_nodes: range of arguments of interpolated function
    :param y_nodes: range of interpolated values
    :param plot_title: title of the plot
    """
    for x_nodes, y_nodes, order in zip(x_nodes_list, y_nodes_list, range(2, 5)):
        plt.plot(x_nodes, y_nodes, label='{0}'.format(order), linewidth=0.3)

    plt.plot(x_nodes_list[-1], [explicit_solution(x) for x in x_nodes_list[-1]])
    plt.title("Runge-Kutta method")

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_nodes_list, y_nodes_list):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot,
                      args=(x_nodes_list, y_nodes_list))
    process.start()


def runge_kutta_interpolation_node(func, x_node, y_node, step, accuracy):
    """
        Value node for Runge-Kutta interpolation
    :param func: given function from the task
    :param x_node: x coordinate of interpolated node
    :param y_node: y coordinate of interpolated node
    :param accurancy: order of accuracy
    :return: interpolated node
    """
    accuracies = [func(x_node, y_node), ]
    for i in range(0, accuracy - 2):
        accuracies.append(2 * func(x_node + step / 2, y_node + step * accuracies[-1]))
    accuracies.append(func(x_node + step, y_node + step * accuracies[-1]))

    delimer = {2: 2, 3: 4, 4: 6}
    return y_node + (step / delimer[accuracy]) * sum(accuracies)


def runge_kutta_interpolation(func, x_0, y_0, segment_end, step, accuracy):
    """
        Classic Runge-Kutta method for solving ODE
    :param func: given function from the task
    :param x_0: initial argument
    :param y_0: initial value
    :param segment_end: end of interpolation segment
    :param step: interpolation accurancy
    :return: interpolated nodes
    """
    x_nodes = np.arange(x_0, segment_end, step)
    y_nodes = [y_0, ]
    for x_node in x_nodes[1:]:
        y_nodes.append(runge_kutta_interpolation_node(func, x_node, y_nodes[-1], step, accuracy))

    return x_nodes, y_nodes


def given_function(x, y):
    """ Given function from the task """
    return x**2 + 2 * x + y / (x + 2)


def explicit_solution(x):
    """ Explicit solution of the given task """
    return 0.5 * (x**3 + 2 * x**2 + 1.51 * x + 3.02)


def main():
    """ Execution logic """
    x_0, y_0 = 0.5, 2.2
    segment_end = 5.5
    step = 0.1

    x_nodes_list, y_nodes_list = [], []
    for accuracy in range(2, 5):
        x_nodes, y_nodes = runge_kutta_interpolation(given_function, x_0, y_0, segment_end, step, accuracy)
        x_nodes_list.append(x_nodes)
        y_nodes_list.append(y_nodes)

    display_plot_async(x_nodes_list, y_nodes_list)

    print('Runge-Kutta interpolation results')
    print(tp.header(['X', 'Y', 'Y2', 'Y3', 'Y4']))
    for x, y, y2, y3, y4 in zip(x_nodes_list[0], [explicit_solution(x) for x in x_nodes_list[0]],
                                y_nodes_list[0], y_nodes_list[1], y_nodes_list[2]):
        print(tp.row([x, y, y2, y3, y4, ]))


if __name__ == '__main__':
    main()
