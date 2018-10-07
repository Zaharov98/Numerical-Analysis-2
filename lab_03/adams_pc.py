"""
    class contains hardcoded solution for Adams prognosis corrector method
"""

import numpy as np
import euler_pc


class AdamsPrognosisCorrector:
    def solve(self, func, x_0, y_0, segment_end, step, accuracy):
        if accuracy == 2:
            integrated_lagrange = lambda x, y: (3 / 2 * func(x[1], y[1]) - 1 / 2 * func(x[0], y[0])) * step
        elif accuracy == 3:
            integrated_lagrange = lambda x, y: (23 / 12 * func(x[2], y[2]) - 4 / 3 * func(x[1], y[1])
                                                + 5 / 12 * func(x[0], y[0])) * step
        elif accuracy == 4:
            integrated_lagrange = lambda x, y: (55 / 24 * func(x[3], y[3]) - 59 / 24 * func(x[2], y[2])
                                                + 37 / 24 * func(x[1], y[1]) - 3 / 8 * func(x[0], y[0])) * step
        else:
            raise ValueError('Invalid accuracy')

        return self._adams_prognosis_correction_method(func, x_0, y_0, segment_end, step, accuracy, integrated_lagrange)

    def _adams_prognosis_correction_method(self, func, x_0, y_0, segment_end, step, accuracy, integrated_lagrange):
        initial_segment = x_0 + step * accuracy
        x_nodes, y_nodes = euler_pc.euler_predictor_corrector_method(func, x_0, y_0, initial_segment, step, accuracy=4)

        while x_nodes[-1] <= segment_end:
            y_temp = y_nodes[-1] + integrated_lagrange(x_nodes[-accuracy:], y_nodes[-accuracy:])
            y_node = y_nodes[-1] + integrated_lagrange(x_nodes[-accuracy:],
                                                       np.append(y_nodes[-accuracy: -1], [y_temp, ]))

            x_nodes = np.append(x_nodes, [x_nodes[-1] + step, ])
            y_nodes = np.append(y_nodes, [y_node, ])

        return x_nodes, y_nodes
