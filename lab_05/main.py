"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 6, Zaharov Igor

    Finite difference method
"""

import math
import numpy.linalg
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from boundary_problem import BoundaryProblem


def get_boundary_problem_matrix(x, problem: BoundaryProblem ):
    return 0


def main():
    problem = BoundaryProblem(a=[1, 0], b=[2, 3], A=0.5, B=1.2)

    left, right, step = 0.5, 1.5, 0.1
    x = np.arange(left, right, step)

    matrix = get_boundary_problem_matrix(x, problem)
    # solve matrix equation using numpy.linalg.solve
    # print x, and y shapes by pyplot, tableprint


if __name__ == '__main__':
    main()
