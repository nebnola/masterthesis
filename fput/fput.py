import numpy as np

def fput_ode(t, y: np.ndarray, alpha, m=1,):
    """
    Right hand side of the first-order ODE for the FPUT system

    Args:
        y: A vector of the form (x1, p1, x2, p2, ...)
        alpha: the size of the nonlinearity
        m: the mass of the oscillators

    Returns:
        The derivative of y
    """
    y = y.reshape((-1, 2))
    x, p = y[:, 0], y[:, 1]

    x_right = np.roll(x, -1)
    x_left = np.roll(x, 1)
    dp = (x_right - x) + (x_left - x) + alpha * (x_right - x)**2 - alpha * (x_left - x)**2

    dx = p/m
    dy = np.stack((dx, dp), axis=1)
    return dy.flatten()
