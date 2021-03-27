from math import tanh
import numpy as np
from prettytable import PrettyTable

def H(x, y):
    return tanh(x * y) / 2.

def f(x):
    return x - 0.5

def simpson(a, b, n):
    step = (b - a) / (2. * n)

    return (np.hstack((np.array([1], float), np.array([4 if i % 2 == 0 else 2 for i in range(2 * n - 1)], float),
                       np.array([1], float))) * step / 3.,
            np.array([a + i * step for i in range(2 * n + 1)], float))

def mechanicQuadratures(coeffs, knots):
    n = len(coeffs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = (1 if i == j else 0) - coeffs[j] * H(knots[i], knots[j])
    g = f(knots)
    z = np.linalg.solve(D, g)

    def u(x):
        return np.sum(coeffs * np.array([H(x, knots[i]) for i in range(len(knots))], float) * z) - f(x)

    return u

if __name__ == "__main__":
    n = 5
    a, b = 0, 1

    eps = 0.000001

    table = PrettyTable()
    table.field_names = ["i", "A_i", "x_i", "u(x_i)"]
    table.align["i"] = "l"
    table.align["A_i"] = "l"
    table.align["x_i"] = "l"
    table.align["u(x_i)"] = "l"

    statistics = PrettyTable()
    statistics.field_names = ["Iteration", "u(a)", "u((a + b) / 2)", "u(b)"]
    statistics.align["Iteration"] = "l"
    statistics.align["u(a)"] = "l"
    statistics.align["u((a + b) / 2)"] = "l"
    statistics.align["u(b)"] = "l"

    iteration = 1

    coeffs, knots = simpson(a, b, n)
    u = mechanicQuadratures(coeffs, knots)
    for i in range(len(coeffs)):
        table.add_row([i, coeffs[i], knots[i], u(knots[i])])
    statistics.add_row([iteration, u(a), u((a + b) / 2.), u(b)])

    print("Iteration:", iteration)
    iteration += 1
    print(table)

    changes = np.array([])

    while True:
        n *= 2
        coeffs, knots = simpson(a, b, n)
        u_new = mechanicQuadratures(coeffs, knots)
        table.clear_rows()
        for i in range(len(coeffs)):
            table.add_row([i, coeffs[i], knots[i], u_new(knots[i])])
        statistics.add_row([iteration, u_new(a), u_new((a + b) / 2.), u_new(b)])

        print("Iteration:", iteration)
        iteration += 1
        print(table)

        changes = np.array([u_new(a + i * (b - a) / 2.) - u(a + i * (b - a) / 2.) for i in range(3)], float)

        if np.max(np.abs(changes)) < eps:
            break

        u = u_new

    print("Statistics:")
    print(statistics)
    print("u_last - u_last_but_one:", changes)
