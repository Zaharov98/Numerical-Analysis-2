
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt


def display_plot(x, t, table: np.matrix):
    colourer = lambda val: clr.to_rgb((val, 0.1, 0.1))
    vfunc = np.vectorize(colourer)

    coloure_scheme = vfunc(table.copy())

    plt.scatter(x, t, s=(len(x), len(t)), c=coloure_scheme)
    plt.show()



