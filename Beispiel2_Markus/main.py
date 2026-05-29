from integrate import T2, T6, T8
import numpy as np
import matplotlib.pyplot as plt
import os


EPS = 2.22045e-16
TOL = 100*EPS


class Integral:
    minimum_increment = 0

    @staticmethod
    def get_min_n(min_n):
        print(min_n)
        raise NotImplementedError

    @staticmethod
    def integrate(func, a, b, n):
        print(func, a, b, n)
        raise NotImplementedError


class Trapez(Integral):
    minimum_increment = 1

    @staticmethod
    def get_min_n(min_n):
        if min_n > 1:
            return min_n
        return 2

    @staticmethod
    def integrate(func, a, b, n):
        return T2(func, a, b, n)


class Milne(Integral):
    minimum_increment = 4

    @staticmethod
    def get_min_n(min_n):
        return 1 + min_n + min_n%Milne.minimum_increment

    @staticmethod
    def integrate(func, a, b, n):
        return T6(func, a, b, n)


class Weddler(Integral):
    minimum_increment = 6

    @staticmethod
    def get_min_n(min_n):
        return 1 + min_n + min_n%Milne.minimum_increment

    @staticmethod
    def integrate(func, a, b, n):
        return T8(func, a, b, n)


class TestCase:
    def __init__(self, func, a, b, solution, label) -> None:
        self.func = func
        self.a = a
        self.b = b
        self.solution = solution
        self.label = label


def draw_tol(plotter):
    plotter.axhline(y=TOL, color='gray', linestyle='--', label="$100\\cdot\\text{eps}$")


def add_threshold_ticks(ax, ns, errors, threshold):
    cross = next((n for n, e in zip(ns, errors) if e < threshold), None)
    if cross is not None:
        ticks = [t for t in ax.get_xticks() if abs(t - cross) > 10]
        ax.set_xticks(sorted(set(ticks + [cross])))
    return cross


def test(method: Integral, test_case: TestCase, min_n=10, max_n = 500):
    abs_error = []
    N = []

    for n in range(method.get_min_n(min_n), max_n, method.minimum_increment):
        N.append(n)
        abs_error.append(
            abs(
                method.integrate(
                    test_case.func,
                    test_case.a,
                    test_case.b,
                    n) - test_case.solution)
        )
    return N, abs_error


def test_function(ax, test_case):
    draw_tol(ax)
    ax.set_title(test_case.label)

    trapez_n, trapez_e   = test(Trapez(),   test_case)
    milne_n,  milne_e    = test(Milne(),    test_case)
    weddler_n, weddler_e = test(Weddler(),  test_case)

    ax.plot(trapez_n,  trapez_e,  label="Trapez")
    ax.plot(milne_n,   milne_e,   label="Milne")
    ax.plot(weddler_n, weddler_e, label="Weddler")

    add_threshold_ticks(ax, trapez_n,  trapez_e,  TOL)
    add_threshold_ticks(ax, milne_n,   milne_e,   TOL)
    add_threshold_ticks(ax, weddler_n, weddler_e, TOL)

    ax.set_yscale("log")
    ax.set_xlim((0, 500))
    ax.legend()


def main():
    plot_dir = "plots"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    glatt = [
        TestCase(np.sin,                        0, np.pi,   2, r"$\sin(x)$ auf $[0,\pi]$"),
        TestCase(np.exp,                        0, 1,       np.e - 1,                 r"$e^x$ auf $[0,1]$"),
        TestCase(np.cos,                        0, np.pi/2, 1, r"$\cos(x)$ auf $[0,\pi/2]$"),
        TestCase(lambda x: 1/(1 + x**2),        0, 1,       np.pi/4, r"$\frac{1}{1+x^2}$ auf $[0,1]$"),
        TestCase(lambda x: np.sin(x)*np.exp(x), 0, np.pi,   (np.exp(np.pi) + 1)/2, r"$\sin(x)\cdot e^x$ auf $[0,\pi]$"),
        TestCase(lambda x: x**5 - 3*x**3 + 2*x, 0, 2,       16/6,     r"$x^5 - 3x^3 + 2x$ auf $[0,2]$"),
    ]
    weniger_glatt = [
        TestCase(np.sqrt,                                            1e-10, 1,
                 2/3,                  r"$\sqrt{x}$ auf $[0,1]$, $f'(0)=\infty$"),
        TestCase(np.abs,                                            -1,    1,
                 1,                    r"$|x|$ auf $[-1,1]$, Knick bei $0$"),
        TestCase(lambda x: np.abs(x)**3,                            -1,    1,
                 0.5,                  r"$|x|^3$ auf $[-1,1]$, $f'''$ discts."),
        TestCase(lambda x: x*np.log(np.where(x > 0, x, 1e-300)),   1e-10, 1,
                 -0.25,                 r"$x\ln(x)$ auf $[0,1]$, $f'(0)=-\infty$"),
        TestCase(lambda x: 1/(1 + 25*x**2),                        -1,    1,
                 2/5*np.arctan(5),     r"$\frac{1}{1+25x^2}$ auf $[-1,1]$, Runge"),
        TestCase(lambda x: np.sqrt(np.abs(x)),                      -1,    1,
                 4/3,                  r"$\sqrt{|x|}$ auf $[-1,1]$, $+\infty$ Abl."),
    ]

    fig, axs = plt.subplots(nrows=len(glatt)//2, ncols=2, figsize=(14, 10))
    fig.suptitle("Fehlerentwicklung bei glatten Funktionen")
    for i in range(0, len(glatt), 2):
        test_function(axs[i//2, 0], glatt[i])
        test_function(axs[i//2, 1], glatt[i+1])
    fig.supylabel("Absoluter Fehler (logarithmisch)")
    fig.supxlabel("Anzahl der benötigten Funktionsauswertungen $n$")
    fig.tight_layout(rect=(0.01, 0.01, 1, 1))
    plt.savefig(os.path.join(plot_dir, "glatt.png"))
    plt.close()

    fig, axs = plt.subplots(nrows=len(weniger_glatt)//2, ncols=2, figsize=(14, 10))
    fig.suptitle("Fehlerentwicklung bei weniger glatten Funktionen")
    for i in range(0, len(weniger_glatt), 2):
        test_function(axs[i//2, 0], weniger_glatt[i])
        test_function(axs[i//2, 1], weniger_glatt[i+1])
    fig.supylabel("Absoluter Fehler (logarithmisch)")
    fig.supxlabel("Anzahl der benötigten Funktionsauswertungen $n$")
    fig.tight_layout(rect=(0.01, 0.01, 1, 1))
    plt.savefig(os.path.join(plot_dir, "weniger_glatt.png"))
    plt.close()





if __name__ == "__main__":
    main()

