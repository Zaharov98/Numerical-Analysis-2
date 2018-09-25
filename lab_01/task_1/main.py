"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    The Cauchy problems for ordinary differential equation
"""

import math
import numpy as np
import tableprint as tp
import matplotlib.pyplot as plt
from multiprocessing import Process


def _display_plot(x_nodes, y_nodes, plot_title):
    """
     Display the plot of interpolated points using pyplot
    :param x_nodes: range of arguments of interpolated function
    :param y_nodes: range of interpolated values
    :param plot_title: title of the plot
    """
    plt.plot(x_nodes, y_nodes, 'o', label='interpolated', linewidth=0.1)
    plt.plot(x_nodes, [onhand_solution(x) for x in x_nodes], label='Solution')
    plt.title(plot_title)

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_nodes, y_nodes, plot_title):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot, args=(x_nodes, y_nodes, plot_title))
    process.start()


def given_function(x, y):
    """ Function u(x), given from the task
        dy(x)/dx = y / (x + 1) + exp(x)*(x + 1) """
    return y / (x + 1) + math.exp(x) * (x + 1)


def onhand_solution(x):
    """ Solution of given tasks made by the hands """
    return math.exp(x) * (x + 1)


def implicit_euler_interpolation(func, x_0, y_0, segment_start, segment_end, step):
    """
        Implicit interpolation method for ordinary differential equation
    :param func: function derivative
    :param x_0: initial point x coordinate
    :param y_0: initial point y coordinate
    :param segment_start: left border of interpolation segment
    :param segment_end: right border of interpolation segment
    :param step: step for interpolation nodes
    :return: range of x,y coordinates of interpolated function
    """
    x_nodes = np.arange(segment_start, segment_end, step)
    y_nodes = [y_0, ]
    for x_node in x_nodes[1:]:
        y_node = y_nodes[-1] + step * func(x_node, y_nodes[-1])
        y_nodes.append(y_node)

    return x_nodes, np.array(y_nodes)


def cauchy_euler_interpolation(func, x_0, y_0, segment_start, segment_end, step):
    """
        Cauchy-Euler interpolation method for ordinary differential equation
    :param func: function derivative
    :param x_0: initial point x coordinate
    :param y_0: initial point y coordinate
    :param segment_start: left border of interpolation segment
    :param segment_end: right border of interpolation segment
    :param step: step for interpolation nodes
    :return: range of x,y coordinates of interpolated function
    """
    x_nodes = np.arange(segment_start, segment_end, step)
    y_nodes = [y_0, ]
    for x_node in x_nodes[1:]:
        y_node = y_nodes[-1] + step * func(x_node + step / 2, y_nodes[-1] + (step / 2) * func(x_node, y_nodes[-1]))
        y_nodes.append(y_node)

    return x_nodes, np.array(y_nodes)


def run_implicit_euler_method():
    """ Run implicit euler interpolation calculating """
    step = 0.05
    x_0, y_0 = 0, 1
    segment_end = 1

    x_nodes, y_nodes = implicit_euler_interpolation(given_function, x_0, y_0,
                                                    segment_start=x_0, segment_end=segment_end, step=step)
    display_plot_async(x_nodes, y_nodes, "Implicit Eulerian interpolation")

    print('\nExplicit Euler method results')
    print(tp.header(['X', 'Y', 'Interpolated', ]))
    for x, y in zip(x_nodes, y_nodes):
        print(tp.row([x, onhand_solution(x), y]))


def run_cauchy_euler_method():
    """ Run Cauchy-Euler interpolation calculating """
    step = 0.1
    x_0, y_0 = 0, 1
    segment_end = 1

    x_nodes, y_nodes = cauchy_euler_interpolation(given_function, x_0, y_0,
                                                  segment_start=x_0, segment_end=segment_end, step=step)
    display_plot_async(x_nodes, y_nodes, "Cauchy-Euler interpolation")

    print('\nCauchy-Euler method results')
    print(tp.header(['X', 'Y', 'Interpolated', ]))
    for x, y in zip(x_nodes, y_nodes):
        print(tp.row([x, onhand_solution(x), y]))


def main():
    """ Execution logic """
    run_implicit_euler_method()
    run_cauchy_euler_method()


if __name__ == '__main__':
    main()
