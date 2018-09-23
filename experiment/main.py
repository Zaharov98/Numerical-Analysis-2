"""
    Theory of Probability and Mathematical Statistics task
"""

import math
import pprint
import numpy as np
import data.sequence
from plotting import freq_polygon


def main():
    data_row = data.sequence.row
    variation_row = np.array(sorted(data_row))

    swing = variation_row[-1] - variation_row[0]

    interval_numb = 9
    h = swing / interval_numb

    x_min = variation_row[0]
    intervals = [(x_min + h * step, x_min + h * (step + 1))
                 for step in range(0, interval_numb + 1)]

    interval_freq = []
    for interval in intervals:
        freq = len(list(filter(lambda x: interval[0] <= x <= interval[1], variation_row)))
        interval_freq.append(freq)
    interval_middle = [(interval[0] + interval[1]) / 2 for interval in intervals]
    freq_polygon.graph(interval_middle, interval_freq)


if __name__ == '__main__':
    main()
