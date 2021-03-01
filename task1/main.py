import numpy as np

def Gauss(matrix, vector):
    a = matrix.copy()
    b = vector.copy()

    n, _ = a.shape

    result = np.array([0 for i in range(0, n)], float)

    for k in range(0, n):
        p = np.abs(a[k:, k]).argmax() + k
        if p != k:
            a[k], a[p] = a[p].copy(), a[k].copy()
            b[k], b[p] = b[p].copy(), b[k].copy()

        tmp = a[k, k]
        a[k, k + 1:] /= tmp
        b[k] /= tmp

        for i in range(k + 1, n):
            tmp = a[i, k]
            a[i, k + 1:] -= a[k, k + 1:] * tmp
            b[i] -= b[k] * tmp

    for i in range(n - 1, -1, -1):
        sum = np.sum(result[i + 1:] * a[i, i + 1:])
        result[i] = b[i] - sum

    return result

def Jordan(matrix):
    n, _ = matrix.shape
    common = np.hstack((matrix, np.identity(n, float)))

    for k in range(0, n):
        p = np.abs(common[k:, k]).argmax() + k
        if p != k:
            common[k], common[p] = common[p].copy(), common[k].copy()

        tmp = common[k, k]
        common[k, k + 1:] /= tmp

        for i in range(0, n):
            if i == k:
                continue
            tmp = common[i, k]
            common[i, k + 1:] -= common[k, k + 1:] * tmp

    return common[:, n:]

def LU(matrix):
    n, _ = matrix.shape
    l = np.zeros_like(matrix)
    u = np.zeros_like(matrix)

    for i in range(0, n):
        for j in range(i, n):
            l[j, i] = matrix[j, i] - np.sum(l[j, :i] * u[:i, i])
            u[i, j] = (matrix[i, j] - np.sum(l[i, :i] * u[:i, j])) / l[i, i]

    return l, u

def det(matrix):
    l, _ = LU(matrix)
    n, _ = l.shape
    result = 1
    for i in range(0, n):
        result *= l[i, i]
    return result

def cond(matrix):
    return np.linalg.norm(matrix, 2) * np.linalg.norm(Jordan(matrix), 2)

def errorEstimate(matrix, dmatrix, vector, dvector):
    x = np.linalg.solve(matrix, vector)
    dx = x - np.linalg.solve(matrix + dmatrix, vector + dvector)
    error = np.linalg.norm(dx) / np.linalg.norm(x)

    cond_ = cond(matrix)
    errorEstimate = cond_ / (1. - np.linalg.norm(matrix, 2) * np.linalg.norm(dmatrix, 2)) *\
                    (np.linalg.norm(dvector) / np.linalg.norm(vector) + np.linalg.norm(dmatrix, 2) / np.linalg.norm(matrix, 2))

    return error, errorEstimate

a = np.array([[-402.9,200.7],[1204.2,-603.6]], float)
da = np.array([[0,0],[0,0]], float)
b = np.array([200,-600], float)
db = np.array([-1,-1], float)

print("cond(A) = ", cond(a))
error, estimate = errorEstimate(a, da, b, db)
print("Error = ", error)
print("Error estimate = ", estimate)
print("============================")

# Variant 7
a = np.array([[9.331343,1.120045,-2.880925],[1.120045,7.086042,0.670297],[-2.880925,0.670297,5.622534]], float)
b = np.array([7.570463,8.876384,3.411906], float)

print("A =")
print(a)
print("A^(-1) =")
print(Jordan(a))

print()
print("det(A) via LU-decomposition:")
print(det(a))
print("Accurate det(A):")
print(np.linalg.det(a))


print()
print("Solution by Gauss scheme:")
x = Gauss(a, b)
print(x)
print("b - Ax = ")
print(b - a.dot(x))
a[0, 0] *= 0.00000001
print()
print("Solution by Gauss scheme with changed matrix:")
x = Gauss(a, b)
print(x)
print("b - Cx = ")
print(b - a.dot(x))