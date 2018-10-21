"""
    Class representing boundary problem
"""


class BoundaryProblem:
    """ Naming is from the task """
    def __init__(self, a, b, A, B):
        self._a = a
        self._b = b
        self._A = A
        self._B = B

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    def p(self, x):
        return 1 / x

    def q(self, x):
        return 2

    def f(self, x):
        return x
