import numpy as np

def Gauss(matrix, vector):
    n, _ = matrix.shape
    result = np.array([0 for i in range(0, n)], float)

    for k in range(0, n):
        p = np.abs(matrix[k:, k]).argmax() + k
        if p != k:
            matrix[k], matrix[p] = matrix[p].copy(), matrix[k].copy()
            vector[k], vector[p] = vector[p].copy(), vector[k].copy()

        tmp = matrix[k, k]
        matrix[k, k + 1:] /= tmp
        vector[k] /= tmp

        for i in range(k + 1, n):
            tmp = matrix[i, k]
            matrix[i, k + 1:] -= matrix[k, k + 1:] * tmp
            vector[i] -= vector[k] * tmp

    for i in range(n - 1, -1, -1):
        sum = np.sum(result[i + 1:] * matrix[i, i + 1:])
        result[i] = vector[i] - sum

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
    error = abs(dx) / abs(x)

    cond = cond(matrix)
    errorEstimate = cond / (1. - np.linalg.norm(matrix, 2) * np.linalg.norm(dmatrix, 2)) *\
                    (np.linalg.norm(dvector) / np.linalg.norm(vector) + np.linalg.norm(dmatrix, 2) / np.linalg.norm(matrix, 2))

    return error, errorEstimate

m = np.array([[1,0,1,0],[-1,1,-2,1],[4,0,1,-2],[-4,4,0,1]], float)
v = np.array([2,-2,0,5], float)

# print("Answer (Gauss): ", np.linalg.solve(m, v))
# print("My answer (Gauss): ", Gauss(m, v))
# print()
# print("Answer (Jordan): ", np.linalg.inv(m))
# print("My answer (Jordan): ", Jordan(m))
m = np.array([[2,1],[0,5]], float)
v = np.array([3,1,4,1,5], float)

print(det(m))
