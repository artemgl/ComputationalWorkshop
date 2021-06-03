import numpy as np
import math
from prettytable import PrettyTable

def L(u, du, d2u, x):
    return - (4 + x) / (5 + 2 * x) * d2u + (0.5 * x - 1) * du + (1 + math.exp(0.5 * x)) * u

def F(x):
    return 2. + x

def Legendre_with_derivatives(n, x):
    if n == 0:
        return [1], [0]

    meanings = [1, x]
    derivatives = [0, 1]

    for i in range(2, n + 1):
        meanings.append((2 * i - 1) / i * x * meanings[i - 1] - (i - 1) / i * meanings[i - 2])
        derivatives.append(i / (1 - x * x) * (meanings[i - 1] - x * meanings[i]))

    return meanings, derivatives

def Legendre_roots_with_coeffs(n, eps):
    roots = [math.cos((4 * i + 3) / (4 * n + 2) * math.pi) for i in range(n)]
    coeffs = []

    for i in range(n):
        root = roots[i]
        while True:
            meanings, derivatives = Legendre_with_derivatives(n, root)
            next_root = root - meanings[n] / derivatives[n]
            if abs(next_root - root) < eps:
                root = next_root
                break
            root = next_root
        roots[i] = root

        meanings, _ = Legendre_with_derivatives(n - 1, root)
        coeffs.append(2 * (1 - root * root) / (n * n * meanings[n - 1] * meanings[n - 1]))

    return roots, coeffs

def Gauss(f, n, eps):
    roots, coeffs = Legendre_roots_with_coeffs(n, eps)
    result = 0
    for i in range(len(roots)):
        result += coeffs[i] * f(roots[i])
    return result

def dot(f, g, n, eps):
    def fg(x):
        return f(x) * g(x)
    return Gauss(fg, n, eps)

def Jacobi(n, k, x):
    if n < 0:
        return []
    if n == 0:
        return [1]

    result = [1, (k + 1) * x]

    for i in range(n - 1):
        result.append(((i + k + 2) * (2 * i + 2 * k + 3) * x * result[i + 1] - (i + k + 2) * (i + k + 1) * result[i]) / ((i + 2 * k + 2) * (i + 2)))

    return result

def Jacobi_with_derivatives(n, k, x):
    if n == 0:
        return [1], [0], [0]

    meanings = Jacobi(n, k, x)
    derivatives = [0, k + 1]
    derivatives2 = [0, 0]

    meanings_ = Jacobi(n - 1, k + 1, x)
    meanings__ = Jacobi(n - 2, k + 2, x)
    for i in range(2, n + 1):
        derivatives.append(meanings_[i - 1] * (i + 2 * k + 1) * 0.5)
        derivatives2.append(meanings__[i - 2] * (i + 2 * k + 1) * (i + 2 * k + 2) * 0.25)

    return meanings, derivatives, derivatives2

# omega_1(x) = x^2 + 2x - 11
# omega_2(x) = x^3 - 3x + 2
# omega_n(x) = (1 - x^2)^2 * P_{i-2}^{2,2}(x)
def omega_with_derivatives(n, x):
    if n == 0:
        return [x * x + 2 * x - 11], [2 * x + 2], [2]

    meanings, derivatives, derivatives2 = Jacobi_with_derivatives(n - 2, 2, x)

    result_meanings = [x * x + 2 * x - 11, x * x * x - 3 * x + 2]
    result_derivatives = [2 * x + 2, 3 * x * x - 3]
    result_derivatives2 = [2, 6 * x]
    for i in range(2, n + 1):
        p = x * x - 1
        result_meanings.append(p * p * meanings[i - 2])
        result_derivatives.append(4 * p * x * meanings[i - 2] + p * p * derivatives[i - 2])
        result_derivatives2.append(4 * (3 * x * x - 1) * meanings[i - 2] + 8 * p * x * derivatives[i - 2] + p * p * derivatives2[i - 2])

    return result_meanings, result_derivatives, result_derivatives2

def phi(n, x):
    return Jacobi(n, 0, x)

def Chebyshev_knots(n):
    return [math.cos((2 * i + 1) / (2 * n) * math.pi) for i in range(n)]

def fillTable(table, A, f, n):
    for k in range(3, n + 1):
        c = np.linalg.solve(A[:k, :k], f[:k])
        def u(x):
            result = 0
            omega, _, _ = omega_with_derivatives(k - 1, x)
            for i in range(len(omega)):
                result += c[i] * omega[i]
            return result
        table.add_row([k, np.linalg.cond(A[:k, :k]), u(-0.5), u(0), u(0.5)])

if __name__ == "__main__":
    moments = PrettyTable()
    moments.field_names = ["n", "mu(A)", "yn(-0.5)", "yn(0)", "yn(0.5)"]
    moments.align["n"] = "l"
    moments.align["mu(A)"] = "l"
    moments.align["yn(-0.5)"] = "l"
    moments.align["yn(0)"] = "l"
    moments.align["yn(0.5)"] = "l"

    collocation = PrettyTable()
    collocation.field_names = ["n", "mu(A)", "yn(-0.5)", "yn(0)", "yn(0.5)"]
    collocation.align["n"] = "l"
    collocation.align["mu(A)"] = "l"
    collocation.align["yn(-0.5)"] = "l"
    collocation.align["yn(0)"] = "l"
    collocation.align["yn(0.5)"] = "l"

    n = 7
    eps = 0.000001

    A = np.zeros((n, n))
    f = np.zeros(n)
    for i in range(n):
        def g(x):
            phi_ = phi(i, x)
            return phi_[i]
        for j in range(n):
            def h(x):
                omega, domega, d2omega = omega_with_derivatives(j, x)
                return L(omega[j], domega[j], d2omega[j], x)
            A[i, j] = dot(h, g, i + j + 2, eps)
        f[i] = dot(F, g, (i + 1) // 2 + 1, eps)

    fillTable(moments, A, f, n)
    print("Moments")
    print(moments)
    print("Coefficients:")
    print(np.linalg.solve(A, f))

    A = np.zeros((n, n))
    f = np.zeros(n)
    knots = Chebyshev_knots(n)
    for i in range(n):
        x = knots[i]
        for j in range(n):
            omega, domega, d2omega = omega_with_derivatives(j, x)
            A[i, j] = L(omega[j], domega[j], d2omega[j], x)
        f[i] = F(x)

    fillTable(collocation, A, f, n)
    print("Collocation")
    print(collocation)
    print("Coefficients:")
    print(np.linalg.solve(A, f))
