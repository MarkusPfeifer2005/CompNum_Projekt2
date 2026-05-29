from typing import Callable
import numpy as np


def integrate(func, a, b, n):
    """Trapezregel"""
    fx = func(np.linspace(a, b, n))
    s = np.sum(fx[1:-1])
    s += (fx[0] + fx[-1])/2
    h = (b-a)/(n-1)
    return s*h


def T(func: Callable[[np.ndarray], np.ndarray], a: np.float64, b: np.float64, n: int, c:
      np.ndarray) -> np.float64:
    """integrations Backend

    Args:
        func: Funktion die integriert wird
        a: Start des Integrationsintervalls
        b: Ende des Integrationsintervalls
        n: Anzahl der Punkte im Intervall, für die Gewichte benötigt werden (a
           und b sind miteinbeschlossen)
        c: np.ndarray, das die Gewichte beinhaltet

    Returns:
        das berechnete Integral

    Raises:
        AssertionError wenn die Anzahl der mitgelieferten Gewichte nicht zu der
        Anzahl an Schritten passt
    """
    assert (n-1) % (len(c)-1) == 0, f"Die Anzahl der mitgelieferten Gewichte\
    {c = } passt nicht mit der Anzahl der Schritte {n = } zusammen!"
    offset = len(c) -1
    h = (b-a)/(n-1)
    C = np.zeros((n,), dtype=np.float64)
    for i in range((n-1)//offset):
        C[i*offset : i*offset + len(c)] += c
    fx = func(np.linspace(a, b, n, dtype=np.float64))
    return h * np.sum(C * fx)


def T2(func: Callable[[np.ndarray], np.ndarray], a: np.float64, b: np.float64,
       n: int) -> np.float64:
    """Trapezregel für numerische Integration

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


def T6(func: Callable[[np.ndarray], np.ndarray], a: np.float64, b: np.float64,
       n: int) -> np.float64:
    """MilneRegel für numerische Integration

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


def T8(func: Callable[[np.ndarray], np.ndarray], a: np.float64, b: np.float64,
       n: int) -> np.float64:
    """WeddleRegel für numerische Integration

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

