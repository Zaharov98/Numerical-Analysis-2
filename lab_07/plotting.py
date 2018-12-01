
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt


def _heatmap(data, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
        Create a heatmap from a numpy array and two lists of labels.
    :param data: A 2D numpy array of shape (N,M)
    :param row_labels: A list or array of length N with the labels for the rows
    :param col_labels: A list or array of length M with the labels for the columns
    :param ax: A matplotlib.axes.Axes instance to which the heatmap is plotted.
     If not provided, use current axes or create a new one.
    :param cbar_kw:  A dictionary with arguments to :meth:`matplotlib.Figure.colorbar`.
    :param cbarlabel: The label for the colorbar
    :param kwargs: All other arguments are directly passed on to the imshow call.
    """
    fig, ax = plt.subplots()

    im = ax.imshow(data)
    color_bar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    color_bar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    fig.tight_layout()
    plt.show()


def display_plot(x: np.array, t: np.array, table: np.array):
    """
        Display plots related to heatmap
    :param x: np.array of X dimension
    :param t: np.array of Y demension
    :param table: (len(t), len(x)) value map
    """
    _heatmap(table)
