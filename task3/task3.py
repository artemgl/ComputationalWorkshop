import numpy as np
from prettytable import PrettyTable

def posteriorEstimate(a, l, y):
    return np.linalg.norm(a.dot(y) - l * y, 2) / np.linalg.norm(y, 2)

def powMethod(a, y0, eps):
    table = table = PrettyTable()
    table.field_names = ["k", "lambda(k)", "|lambda(k) - lambda(k-1)|", "|lambda(k) - lambda*|", "||Ax(k) - lambda(k)x(k)||",
                         "Error estimate posterior"]
    table.align["lambda(k)"] = "l"
    table.align["|lambda(k) - lambda(k-1)|"] = "l"
    table.align["|lambda(k) - lambda*|"] = "l"
    table.align["||Ax(k) - lambda(k)x(k)||"] = "l"
    table.align["Error estimate posterior"] = "l"

    y = y0.copy()
    l = 0
    iterations = 0
    while True:
        (n,) = y.shape
        p = 0
        maxComponent = 0
        for i in range(0, n):
            if abs(y[i]) > maxComponent:
                maxComponent = abs(y[i])
                p = i
        y /= y[p]

        nextY = a.dot(y)
        lPrev = l
        l = nextY[p]
        iterations += 1
        pe = posteriorEstimate(a, l, y)
        table.add_row([iterations, l, abs(l - lPrev), abs(l - lAcc), np.linalg.norm(a.dot(y) - l * y, 2), pe])
        if pe <= eps:
            break
        y = nextY
    print("Pow method")
    print(table)
    return l, y / np.linalg.norm(y, 2)

def scalMethod(a, y0, eps):
    table = table = PrettyTable()
    table.field_names = ["k", "lambda(k)", "|lambda(k) - lambda(k-1)|", "|lambda(k) - lambda*|", "||Ax(k) - lambda(k)x(k)||",
                         "Error estimate posterior"]
    table.align["lambda(k)"] = "l"
    table.align["|lambda(k) - lambda(k-1)|"] = "l"
    table.align["|lambda(k) - lambda*|"] = "l"
    table.align["||Ax(k) - lambda(k)x(k)||"] = "l"
    table.align["Error estimate posterior"] = "l"

    y = y0.copy() / np.linalg.norm(y0, 2)
    l = 0
    iterations = 0
    while True:
        nextY = a.dot(y)
        lPrev = l
        l = nextY.dot(y) / y.dot(y)
        iterations += 1
        pe = posteriorEstimate(a, l, y)
        table.add_row([iterations, l, abs(l - lPrev), abs(l - lAcc), np.linalg.norm(a.dot(y) - l * y, 2), pe])
        if pe <= eps:
            break
        y = nextY / np.linalg.norm(nextY, 2)
    print("Scalar method")
    print(table)
    return l, y

eps = 0.001
a = np.array([[-1.00449,-0.38726,0.59047],[-0.38726,0.73999,0.12519],[0.59047,0.12519,-1.0866]], float)
y0 = np.array([1,1,1], float)

wa, va = np.linalg.eig(a)
(n,) = wa.shape
lAcc = wa[np.abs(wa).argmax()]

lPow, vPow = powMethod(a, y0, eps)
print()
lScal, vScal = scalMethod(a, y0, eps)
print()

print("Eigenvalues:")
print(wa)
print("Eigenvectors:")
print(va)
waSorted = np.sort(np.abs(wa))
print("lambda_2 / lambda_1 = ", waSorted[n - 2] / waSorted[n - 1])
print()

print("Eigenvalue by pow method:")
print(lPow)
print("Eigenvector by pow method:")
print(vPow)
print()

print("Eigenvalue by scalar method:")
print(lScal)
print("Eigenvector by scalar method:")
print(vScal)
