""" Class representing the system of functions from the task """

import math


class System:
    @staticmethod
    def dy(x, y, z):
        return y + 5 * z

    @staticmethod
    def y(x):
        return math.exp(-x) * (math.cos(x) + 7 * math.sin(x))

    @staticmethod
    def dz(x, y, z):
        return -y - 3 * z

    @staticmethod
    def z(x):
        return math.exp(-x) * (math.cos(x) - 3 * math.sin(x))
