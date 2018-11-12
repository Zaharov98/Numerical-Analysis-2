
import numpy as np


def get_coefficients(h, n):
    coefs = [h / 3, ]
    is_even = lambda numb: numb % 2 == 0

    for i in range(1, n - 1):
        if is_even(i):
            coefs.append(4 * h / 3)
        else:
            coefs.append(2 * h / 3)

    coefs.append(h / 3)
    return np.array(coefs)
