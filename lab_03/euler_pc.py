"""
    Euler predictor corrector method
"""


def euler_predictor_corrector_method(func, x_0, y_0, segment_end, step, accuracy):
    """
        Explicit Euler predictor corrector method
    :param func: given function from the task
    :param x_0: initial argument
    :param y_0: initial value
    :param segment_end: end of the interpolation segment
    :param step: step for x nodes
    :param accuracy: accuracy of the method
    :return: x and y interpolation nodes
    """
    x_nodes = [x_0, ]
    y_nodes = [y_0, ]
    while x_nodes[-1] <= segment_end:
        x_previous, y_previous = x_nodes[-1], y_nodes[-1]
        y_node = y_nodes[-1] + step * func(x_previous, y_previous)
        for attempt in range(1, accuracy + 1):
            y_node = y_previous + (step / 2) * (func(x_previous, y_previous) + func(x_previous + step, y_node))
        x_nodes.append(x_nodes[-1] + step)
        y_nodes.append(y_node)

    return x_nodes, y_nodes
