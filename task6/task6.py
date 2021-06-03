import numpy as np
from prettytable import PrettyTable

def u(x, t, n):
    if n == 0:
        return x + t
    elif n == 1:
        return x * x + t * t
    elif n == 2:
        return x * x * x + t * t * t
    elif n == 3:
        return np.sin(2 * t + 1) * np.cos(2 * x)

    return 0

def f(x, t, n):
    if n == 0:
        # u(x, t) = x + t
        return 0
    elif n == 1:
        # u(x, t) = x^2 + t^2
        return 2 * (t - 2 * x - 1)
    elif n == 2:
        # u(x, t) = x^3 + t^3
        return 3 * (t * t - 3 * x * x - 2 * x)
    elif n == 3:
        # u(x, t) = sin(2t + 1)cos(2x)
        return 2 * (np.cos(2 * t - 2 * x + 1) + 2 * (x + 1) * np.sin(2 * t + 1) * np.cos(2 * x))

    return 0

def phi(x, n):
    if n == 0:
        # u(x, t) = x + t
        return x
    elif n == 1:
        # u(x, t) = x^2 + t^2
        return x * x
    elif n == 2:
        # u(x, t) = x^3 + t^3
        return x * x * x
    elif n == 3:
        # u(x, t) = sin(2t + 1)cos(2x)
        return np.sin(1.) * np.cos(2 * x)

    return 0

def alpha(t, n):
    if n == 0:
        # u(x, t) = x + t
        return 1
    elif n == 1:
        # u(x, t) = x^2 + t^2
        return 0
    elif n == 2:
        # u(x, t) = x^3 + t^3
        return 0
    elif n == 3:
        # u(x, t) = sin(2t + 1)cos(2x)
        return 0

    return 0

def beta(t, n):
    if n == 0:
        # u(x, t) = x + t
        return t + 1
    elif n == 1:
        # u(x, t) = x^2 + t^2
        return t * t + 1
    elif n == 2:
        # u(x, t) = x^3 + t^3
        return t * t * t + 1
    elif n == 3:
        # u(x, t) = sin(2t + 1)cos(2x)
        return np.cos(2.) * np.sin(2 * t + 1)

    return 0

def p(x):
    return 1. + x

def explicit(n, m, T, k_u):
    u = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        u[i, 0] = phi(i / n, k_u)

    def L_h(i, k):
        return p((i + 0.5) / n) * (u[i + 1, k] - u[i, k]) * n * n - p((i - 0.5) / n) * (u[i, k] - u[i - 1, k]) * n * n

    for k in range(1, m + 1):
        for i in range(1, n):
            u[i, k] = u[i, k - 1] + (L_h(i, k - 1) + f(i / n, (k - 1) * T / m, k_u)) * T / m
        u[0, k] = (-2 * alpha(k * T / m, k_u) / n + 4 * u[1, k] - u[2, k]) / 3
        u[n, k] = beta(k * T / m, k_u)

    return u

def weight(w, n, m, T, k_u):
    u = np.zeros((n + 1, m + 1))
    h = 1 / n
    tau = T / m

    for i in range(n + 1):
        u[i, 0] = phi(i / n, k_u)

    def L_h(i, k):
        return p((i + 0.5) / n) * (u[i + 1, k] - u[i, k]) / h / h - p((i - 0.5) / n) * (u[i, k] - u[i - 1, k]) / h / h

    a = [w * p((i + 0.5) / n) / h / h for i in range(n - 1)] + [0]
    b = [n] + [w * (p((i + 0.5) / n) + p((i - 0.5) / n)) / h / h + 1 / tau for i in range(1, n)] + [-1]
    c = [n] + [w * p((i + 0.5) / n) / h / h for i in range(1, n)]

    for k in range(1, m + 1):
        g = [alpha(k * tau, k_u)] +\
            [-1 / tau * u[i, k - 1] - (1 - w) * L_h(i, k - 1) - f(i / n, k * tau - (1 - w) * tau, k_u) for i in range(1, n)] +\
            [beta(k * tau, k_u)]

        u[:, k] = run_method(a, b, c, g)

    return u

def run_method(a, b, c, g):
    s = [c[0] / b[0]]
    t = [-g[0] / b[0]]

    n = len(a)

    for i in range(1, n):
        denominator = b[i] - a[i - 1] * s[i - 1]
        s.append(c[i] / denominator)
        t.append((a[i - 1] * t[i - 1] - g[i]) / denominator)

    y = [(a[n - 1] * t[n - 1] - g[n]) / (b[n] - a[n - 1] * s[n - 1])]

    for i in range(n - 1, -1, -1):
        y.append(s[i] * y[-1] + t[i])

    for i in range(len(y) // 2):
        temp = y[i]
        y[i] = y[-(i + 1)]
        y[-(i + 1)] = temp

    return y

if __name__ == "__main__":
    T = 0.1

    table = PrettyTable()
    table.field_names = ["h", "tau", "|| J_ex - u^(h,tau) ||"]
    table.align["h"] = "l"
    table.align["tau"] = "l"
    table.align["|| J_ex - u^(h,tau) ||"] = "l"

    functions = {0: "x + t", 1: "x^2 + t^2", 2: "x^3 + t^3", 3: "sin(2t + 1)cos(2x)"}

    for k_u in range(4):
        print("u(x, t) = ", functions[k_u], sep='')
        n = 5
        while n <= 20:
            m = 5
            # while m < T * 4 * n * n:
            #     m *= 2

            u_curr = explicit(n, m, T, k_u)
            max1 = 0
            for i in range(n + 1):
                for j in range(m + 1):
                    a = abs(u_curr[i, j] - u(i / n, j * T / m, k_u))
                    if a > max1:
                        max1 = a

            table.add_row([1 / n, T / m, max1])

            n *= 2

        print("Явная схема:")
        print(table)
        table.clear_rows()

        w = 0
        while w <= 1:
            n = 5
            while n <= 20:
                m = 5
                # while m < T * 4 * n * n:
                #     m *= 2

                u_curr = weight(w, n, m, T, k_u)
                max1 = 0
                for i in range(n + 1):
                    for j in range(m + 1):
                        a = abs(u_curr[i, j] - u(i / n, j * T / m, k_u))
                        if a > max1:
                            max1 = a

                table.add_row([1 / n, T / m, max1])

                n *= 2

            print("Схема с весом ", w, ":", sep='')
            print(table)
            table.clear_rows()

            w += 0.5

        print()
        print()
