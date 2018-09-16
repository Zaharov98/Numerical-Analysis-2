"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Adams prognosis-correction method of 2, 3, 4-th orders
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from scipy import interpolate
from scipy import integrate
from runge_kutta_interpolation import runge_kutta_interpolation


def _display_plot(x_nodes_list, y_nodes_list):
    """
     Display the plot of interpolated points using pyplot
    """
    for x_nodes, y_nodes, order in zip(x_nodes_list, y_nodes_list, range(2, 5)):
        plt.plot(x_nodes, y_nodes, label='{0}'.format(order), linewidth=0.3)
    plt.title("Adams prognosis-correction method")

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_nodes_list, y_nodes_list):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot,
                      args=(x_nodes_list, y_nodes_list))
    process.start()


def adam_prognosis_correction_method(func, x_0, y_0, segment_end, step, accuracy=4):
    """

    :param func:
    :param x_0:
    :param y_0:
    :param segment_end:
    :param step:
    :param accuracy:
    :return:
    """
    runge_kutta_interpolation_segment_end = x_0 + step * accuracy
    x_nodes, y_nodes = runge_kutta_interpolation(func, x_0, y_0,
                                                 runge_kutta_interpolation_segment_end, step)

    while x_nodes[-1] <= segment_end:
        func_temp = interpolate.lagrange(x_nodes[-accuracy:], y_nodes[-accuracy:])
        y_temp = y_nodes[-1] + integrate.quad(func_temp, x_nodes[-1], x_nodes[-1] + step)[0]

        func_temp = interpolate.lagrange(x_nodes[-accuracy:], np.append(y_nodes[-accuracy: -1], [y_temp, ]))
        y_node = y_nodes[-1] + integrate.quad(func_temp, x_nodes[-1], x_nodes[-1] + step)[0]

        x_nodes = np.append(x_nodes, [x_nodes[-1] + step])
        y_nodes = np.append(y_nodes, [y_node, ])

    return x_nodes, y_nodes


def given_function(x, y):
    """ Given function from the task """
    return x**2 + 2 * x + y / (x + 2)


def main():
    """ Execution logic """
    x_0, y_0 = 0.5, 2.2
    segment_end = 5.5
    step = 0.1

    x_nodes_list, y_nodes_list = [], []
    for accuracy in range(2, 5):
        x_nodes, y_nodes = adam_prognosis_correction_method(given_function, x_0, y_0, segment_end, step, accuracy)

        x_nodes_list.append(x_nodes)
        y_nodes_list.append(y_nodes)

    display_plot_async(x_nodes_list, y_nodes_list)


if __name__ == '__main__':
    main()
