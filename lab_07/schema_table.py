
import math
import numpy as np


def u_0_t(t: float):
    """ u(0, t) """
    return math.exp(-0.5 * t)


def u_x_0(x: float):
    """ u(x, 0) """
    return math.sin(x)


def u_1_t(t: float):
    """ u(1, t) """
    return - math.exp(-0.5 * t)


def get_schema_table(x: np.array, t: np.array) -> np.array:
    """
    :param x: np.array of x values
    :param t: np.array of t values
    :return: filled solution schema table
    """
    table = _get_initial_table(x, t)
    # i - is for X dimension and j - is for T dimension
    # so for the consistency with formulas rows are iterated by the j
    # and columns iterated by the i
    for j in range(0, len(t) - 1):
        for i in range(1, len(x) - 1):
            table[j + 1][i] = 1 / 2 * (table[j][i + 1] + table[j][i - 1])

    return table


def _get_initial_table(x: np.array, t: np.array) -> np.array:
    """
    :return: table prepared for main calculation
    """
    table = np.zeros(shape=(len(t), len(x)))
    table[0] = np.array(list(map(u_x_0, x)))
    for j in range(0, len(t)):
        table[j][0] = u_0_t(t[j])
    for j in range(0, len(t)):
        table[j][-1] = u_1_t(t[j])

    return table
