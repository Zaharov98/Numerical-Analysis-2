"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    The Cauchy problems for system of ordinary differential equation
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from system import System


def _display_plot(x_range, y_range, x_nodes, y_nodes, plot_title):
    """
     Display the plot of interpolated points using pyplot
    :param x_range: initial function arguments
    :param y_range: initial function values
    :param x_nodes: range of arguments of interpolated function
    :param y_nodes: range of interpolated values
    :param plot_title: title of the plot
    """
    plt.plot(x_nodes, y_nodes, 'o', label='interpolated', linewidth=0.1)
    plt.plot(x_range, y_range, label='initial')
    plt.title(plot_title)

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_range, y_range, x_nodes, y_nodes, plot_title):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot,
                      args=(x_range, y_range, x_nodes, y_nodes, plot_title))
    process.start()


def explicit_euler_method(system, x_0, y_0, z_0, segment_start, segment_end, step):
    """
        Implicit Euler method for system of ordinary differential equations
    :param system: object, representing ODE system
    :param x_0: initial argument
    :param y_0: initial y value
    :param z_0: initial z value
    :param segment_start: left border of interpolated segment
    :param segment_end: right border of interpolated sengent
    :param step: delta for interpolation arguments
    :return: interpolated function nodes
    """
    x_nodes = np.arange(segment_start, segment_end, step)
    y_nodes = [y_0, ]
    z_nodes = [z_0, ]
    for x_node in x_nodes[1:]:
        y_node = y_nodes[-1] + step * system.dy(x_node, y_nodes[-1], z_nodes[-1])
        z_node = z_nodes[-1] + step * system.dz(x_node, y_nodes[-1], z_nodes[-1])

        y_nodes.append(y_node)
        z_nodes.append(z_node)

    return x_nodes, np.array(y_nodes), np.array(z_nodes)


def cauchy_euler_mathod(system, x_0, y_0, z_0, segment_end, step):
    """
        Implicit Euler method for system of ordinary differential equations
    :param system: object, representing ODE system
    :param x_0: initial argument
    :param y_0: initial y value
    :param z_0: initial z value
    :param segment_end: right border of interpolated sengent
    :param step: delta for interpolation arguments
    :return: interpolated function nodes
    """
    x_nodes = [x_0, ]
    y_nodes = [y_0, ]
    z_nodes = [z_0, ]

    while (x_nodes[-1] + step) <= segment_end:
        y_temp = y_nodes[-1] + step * system.dy(x_nodes[-1], y_nodes[-1], z_nodes[-1])
        z_temp = z_nodes[-1] + step * system.dz(x_nodes[-1], y_nodes[-1], z_nodes[-1])

        x_nodes.append(x_nodes[-1] + step)
        y_node = y_nodes[-1] + (step / 2) * (system.dy(x_nodes[-2], y_nodes[-1], z_nodes[-1]) +
                                             system.dy(x_nodes[-1], y_temp, z_temp))
        z_node = z_nodes[-1] + (step / 2) * (system.dz(x_nodes[-2], y_nodes[-1], z_nodes[-1]) +
                                             system.dz(x_nodes[-1], y_temp, z_temp))
        y_nodes.append(y_node)
        z_nodes.append(z_node)

    return x_nodes, y_nodes, z_nodes


def run_explicit_euler_method(steps_numb):
    """ Run implicit Eulerian interpolation """
    x_0, y_0, z_0 = 0, 1, 1
    segment_end = 1
    step = (segment_end - x_0) / steps_numb

    x_nodes, y_nodes, z_nodes = explicit_euler_method(System, x_0, y_0, z_0,
                                                      segment_start=x_0, segment_end=segment_end, step=step)
    x_range = np.arange(x_0, segment_end, 0.05)
    y_range = np.array([System.y(x) for x in x_range])
    z_range = np.array([System.z(x) for x in x_range])

    display_plot_async(x_range, y_range, x_nodes, y_nodes, 'Explicit: Y(x) function for {0} steps'.format(steps_numb))
    display_plot_async(x_range, z_range, x_nodes, z_nodes, 'Explicit: Z(x) function for {0} steps'.format(steps_numb))


def run_cauchy_euler_method(steps_numb):
    """ Run Cauchy-Euler interpolation """
    x_0, y_0, z_0 = 0, 1, 1
    segment_end = 1
    step = (segment_end - x_0) / steps_numb

    x_nodes, y_nodes, z_nodes = cauchy_euler_mathod(System, x_0, y_0, z_0,
                                                    segment_end=segment_end, step=step)
    x_range = np.arange(x_0, segment_end, 0.05)
    y_range = np.array([System.y(x) for x in x_range])
    z_range = np.array([System.z(x) for x in x_range])

    display_plot_async(x_range, y_range, x_nodes, y_nodes, 'Cauchy: Y(x) function for {0} steps'.format(steps_numb))
    display_plot_async(x_range, z_range, x_nodes, z_nodes, 'Cauchy: Z(x) function for {0} steps'.format(steps_numb))


def main():
    """ Execution logic """
    run_explicit_euler_method(steps_numb=10)
    run_cauchy_euler_method(steps_numb=10)


if __name__ == '__main__':
    main()
