"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Adams prognosis-correction method of 2, 3, 4-th orders
"""

import numpy as np
import tableprint as tp
import matplotlib.pyplot as plt
from multiprocessing import Process
from scipy import interpolate, integrate


def _display_plot(x_nodes_list, y_nodes_list):
    """
     Display the plot of interpolated points using pyplot
    """
    for x_nodes, y_nodes, order in zip(x_nodes_list, y_nodes_list, range(2, 5)):
        plt.plot(x_nodes, y_nodes, label='{0}'.format(order), linewidth=0.3)
    plt.plot(x_nodes_list[-1], [explicit_solution(x) for x in x_nodes_list[-1]], linewidth=0.4)
    plt.title('Adams prognosis-correction method')

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def display_plot_async(x_nodes_list, y_nodes_list):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot,
                      args=(x_nodes_list, y_nodes_list))
    process.start()


def adams_prognosis_correction_method(func, x_0, y_0, segment_end, step, accuracy):
    """
        Adams predictor corrector method
    :param func: given function from the task
    :param x_0: initial argument
    :param y_0: initial value
    :param segment_end: end of the interpolation segment
    :param step: step for x nodes
    :param accuracy: accuracy of the method
    :return: x and y interpolation nodes
    """
    initial_segment_end = x_0 + step * accuracy
    x_nodes, y_nodes = euler_predictor_corrector_method(func, x_0, y_0, initial_segment_end, step, accuracy=4)

    while x_nodes[-1] <= segment_end:
        lagrange_pol = interpolate.lagrange(x_nodes[-accuracy:], y_nodes[-accuracy:])
        lagrange_pol_integral = integrate.quad(lagrange_pol, x_nodes[-1], x_nodes[-1] + step)
        y_temp = y_nodes[-1] + lagrange_pol_integral[0]

        lagrange_pol = interpolate.lagrange(x_nodes[-accuracy:], np.append(y_nodes[-accuracy: -1], [y_temp, ]))
        lagrange_pol_integral = integrate.quad(lagrange_pol, x_nodes[-1], x_nodes[-1] + step)
        y_node = y_nodes[-1] + lagrange_pol_integral[0]

        x_nodes = np.append(x_nodes, [x_nodes[-1] + step, ])
        y_nodes = np.append(y_nodes, [y_node, ])

    return x_nodes, y_nodes


def euler_predictor_corrector_method(func, x_0, y_0, segment_end, step, accuracy):
    """
        Explicit Euler predictor corrector method
    :param func: given function from the task
    :param x_0: initial argument
    :param y_0: initial value
    :param segment_end: end of the interpolation segment
    :param step: step for x nodes
    :param accuracy: accuracy of the method
    :return: x and y interpolation nodes
    """
    x_nodes = [x_0, ]
    y_nodes = [y_0, ]
    while x_nodes[-1] <= segment_end:
        x_previous, y_previous = x_nodes[-1], y_nodes[-1]
        y_node = y_nodes[-1] + step * func(x_previous, y_previous)
        for attempt in range(1, accuracy + 1):
            y_node = y_previous + (step / 2) * (func(x_previous, y_previous) + func(x_previous + step, y_node))
        x_nodes.append(x_nodes[-1] + step)
        y_nodes.append(y_node)

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
        x_nodes, y_nodes = adams_prognosis_correction_method(given_function, x_0, y_0, segment_end, step, accuracy)

        x_nodes_list.append(x_nodes)
        y_nodes_list.append(y_nodes)

    display_plot_async(x_nodes_list, y_nodes_list)

    print('Adams predictor-corrector interpolation results')
    print(tp.header(['X', 'Y', 'Y2', 'Y3', 'Y4']))
    for x, y, y2, y3, y4 in zip(x_nodes_list[0], [explicit_solution(x) for x in x_nodes_list[0]],
                                y_nodes_list[0], y_nodes_list[1], y_nodes_list[2]):
        print(tp.row([x, y, y2, y3, y4, ]))


if __name__ == '__main__':
    main()
