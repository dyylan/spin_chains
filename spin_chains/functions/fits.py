import numpy as np
import scipy.special


def inverse_power_fit(x, a, b, c):
    return (a / np.power(x, b)) + c


def yukawa_inverse_power_fit(x, a, b, c, d):
    return [(a * np.exp(-b * i) / np.power(i, c)) + d for i in x]


def power_fit(x, a, b, c):
    return a * np.power(x, b) + c


def sqrt_power_fit(x, a, c):
    return a * np.power(x, 0.5) + c


def one_over_log(x, a):
    return a / np.log(x)


def log(x, a):
    return a + (b * np.log(x))
