"""
    Numeric analysis methods
    Laboratory work number 6
    Variant 6, Zaharov Igor

    Solution of the heat equation
    by the finite difference method
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import schema_table
import plotting


def q(x, t):
    return 0.5 * math.exp(- 0.5 * t) * math.cos(x)


def main():
    compact = 0, 1
    h = 0.01
    x = np.arange(compact[0], compact[1], h)

    et = h**2 / 2
    t = np.arange(compact[0], compact[1], et)

    table = schema_table.get_schema_table(x, t, h, et)
    plotting.display_plot(x, t, table)


if __name__ == '__main__':
    main()
