import numpy as np


def integrate(func, a, b, n):
    fx = func(np.linspace(a, b, n))
    s = np.sum(fx[1:-1])
    s += (fx[0] + fx[-1])/2
    h = (b-a)/(n-1)
    return s*h


def T(func, a, b, n, c):
    assert (n-1) % (len(c)-1) == 0
    offset = len(c) -1
    h = (b-a)/(n-1)
    C = np.zeros((n,), dtype=np.float64)
    for i in range((n-1)//offset):
        C[i*offset : i*offset + len(c)] += c
    fx = func(np.linspace(a, b, n, dtype=np.float64))
    return h * np.sum(C * fx)


def T2(func, a, b, n):
    """das selbe wie integrate"""
    c = np.array([.5, .5], dtype=np.float64)
    return T(func, a, b, n, c)


def T6(func, a, b, n):
    c = np.array([14,64,24,64,14], dtype=np.float64) / 45
    return T(func, a, b, n, c)


def T8(func, a, b, n):
    c = np.array([41,216,27,272,27,216,41], dtype=np.float64)/140
    return T(func, a, b, n, c)

