"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Classical Runge-Kutta method of the n-th order
"""

import numpy as np


def _runge_kutta_interpolation_node(func, x_node, y_node, step, accuracy):
    """
        Value node for Runge-Kutta interpolation
    :param func: given function from the task
    :param x_node: x coordinate of interpolated node
    :param y_node: y coordinate of interpolated node
    :param accuracy: order of accuracy
    :return: interpolated node
    """
    accuracies = [func(x_node, y_node), ]
    for i in range(0, accuracy - 2):
        accuracies.append(2 * func(x_node + step / 2, y_node + step * accuracies[-1]))
    accuracies.append(func(x_node + step, y_node + step * accuracies[-1]))

    delimiter = {2: 2, 3: 4, 4: 6}
    return y_node + (step / delimiter[accuracy]) * sum(accuracies)


def runge_kutta_interpolation(func, x_0, y_0, segment_end, step, accuracy=4):
    """
        Classic Runge-Kutta method for solving ODE
    :param func: given function from the task
    :param x_0: initial argument
    :param y_0: initial value
    :param segment_end: end of interpolation segment
    :param step: interpolation accuracy
    :param accuracy: indicates accuracy order of interpolation
    :return: interpolated nodes
    """
    x_nodes = np.arange(x_0, segment_end, step)
    y_nodes = [y_0, ]
    for x_node in x_nodes[1:]:
        y_nodes.append(_runge_kutta_interpolation_node(func, x_node, y_nodes[-1], step, accuracy))

    return x_nodes, y_nodes
