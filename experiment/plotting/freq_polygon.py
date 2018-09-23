"""
    Frequency polygon
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def _display_plot(x_nodes, y_nodes):
    """
     Display the plot of interpolated points using pyplot
    """

    plt.plot(x_nodes, y_nodes, 'bo', x_nodes, y_nodes, 'k', linewidth=0.3)
    plt.title('Frequency polygon')

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def _display_plot_async(x_nodes, y_nodes):
    """ Run _display_plot in another process """
    process = Process(target=_display_plot,
                      args=(x_nodes, y_nodes))
    process.start()


def graph(x_nodes, y_nodes):
    """ Display the frequency polygon of given values """
    _display_plot_async(x_nodes, y_nodes)
