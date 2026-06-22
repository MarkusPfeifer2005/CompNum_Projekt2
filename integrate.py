from typing import Callable
import numpy as np


def T(func: Callable[[np.ndarray], np.ndarray],
      a: np.float64,
      b: np.float64,
      n: int,
      c: np.ndarray) -> np.float64:
    """
    Args:
        func: Funktion die integriert wird
        a: Start des Integrationsintervalls
        b: Ende des Integrationsintervalls
        n: Anzahl der Punkte im Intervall, fuer die Gewichte benoetigt werden
           (a und b sind miteinbeschlossen)
        c: np.ndarray, das die Gewichte beinhaltet

    Returns:
        das berechnete Integral

    Raises:
        AssertionError wenn die Anzahl der mitgelieferten Gewichte nicht zu der
        Anzahl an Schritten passt
    """
    len_c = len(c)
    offset = len_c - 1

    assert (n-1) % offset == 0, f"Die Anzahl der mitgelieferten Gewichte\
    {c = } passt nicht mit der Anzahl der Schritte {n = } zusammen!"

    h = (b-a)/(n-1)
    C = np.zeros(n, dtype=np.float64)
    for i in range(0, n-1, offset):
        C[i:i+len_c] += c
    fx = func(np.linspace(a, b, n, dtype=np.float64))
    return h * np.sum(C * fx)


def T2(func: Callable[[np.ndarray], np.ndarray],
       a: np.float64,
       b: np.float64,
       n: int) -> np.float64:
    """Trapezregel fuer numerische Integration

    Args:
        func: Funktion die integriert wird
        a: Start des Integrationsintervalls
        b: Ende des Integrationsintervalls
        n: Anzahl der Punkte im Intervall

    Returns:
        das berechnete Integral nach der Trapezregel
    """
    c = np.array([.5, .5], dtype=np.float64)
    return T(func, a, b, n, c)


def T6(func: Callable[[np.ndarray], np.ndarray],
       a: np.float64,
       b: np.float64,
       n: int) -> np.float64:
    """MilneRegel fuer numerische Integration

    Args:
        func: Funktion die integriert wird
        a: Start des Integrationsintervalls
        b: Ende des Integrationsintervalls
        n: Anzahl der Punkte im Intervall

    Returns:
        das berechnete Integral nach der MilneRegel
    """
    c = np.array([14,64,24,64,14], dtype=np.float64) / 45
    return T(func, a, b, n, c)


def T8(func: Callable[[np.ndarray], np.ndarray],
       a: np.float64,
       b: np.float64,
       n: int) -> np.float64:
    """WeddleRegel fuer numerische Integration

    Args:
        func: Funktion die integriert wird
        a: Start des Integrationsintervalls
        b: Ende des Integrationsintervalls
        n: Anzahl der Punkte im Intervall

    Returns:
        das berechnete Integral nach der WeddleRegel
    """
    c = np.array([41,216,27,272,27,216,41], dtype=np.float64)/140
    return T(func, a, b, n, c)


def adaptive_integration(func: Callable[[np.ndarray], np.ndarray],
                         a: np.float64,
                         b: np.float64,
                         TOL: np.float64,
                         c: np.float64 = np.float64(.5)):
    q = 4
    n = 2  # Anzahl der Intervalle
    while True:
        Anzahl_Stuetzstellen = n+1
        xi = np.linspace(a, b, Anzahl_Stuetzstellen, dtype=np.float64)
        fx = func(xi)

        h2 = (b-a)/n
        C = np.ones(fx.shape, np.float64)
        C[0] -= .5
        C[-1] -= .5
        T_h2 = h2 * np.sum(C*fx)
        T_h = h2*2 * np.sum(C[::2] * fx[::2])

        abs_err = np.abs((T_h - T_h2) / 3)
        yield T_h2, Anzahl_Stuetzstellen, abs_err

        if abs_err <= c*TOL:
            break

        n_new = n * (abs_err / (c*TOL))**(1/q)
        n = int(np.ceil(n_new))

