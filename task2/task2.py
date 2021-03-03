import numpy as np
from math import sqrt

def priorEstimate(h, g, x0, k):
    norm = np.linalg.norm(h, np.inf)
    hk = 1
    for i in range(0, k):
        hk *= norm
    return hk * (np.linalg.norm(x0, np.inf) + np.linalg.norm(g, np.inf) / (1 - norm))

def iterations(h, g, x0, k):
    result = [x0]
    for i in range(0, k):
        result.append(h.dot(result[i]) + g)
    return result

def seidel(h, g, x0, k):
    (n,) = x0.shape
    result = [x0]
    for i in range(0, k):
        prev = result[i]
        current = np.zeros(n, float)
        for j in range(0, n):
            v = np.hstack((current[:j], prev[j:]))
            current[j] = h[j].dot(v) + g[j]
        result.append(current)

    return result

def spectralRadius(m):
    wa, _ = np.linalg.eig(m)
    return np.max(np.abs(wa))

def upperRelaxation(h, g, x0, k):
    (n,) = x0.shape

    q = 2 / (1 + sqrt(1 - spectralRadius(h)))

    result = [x0]
    for i in range(0, k):
        prev = result[i]
        current = np.zeros(n, float)
        for j in range(0, n):
            current[j] = prev[j] + q * (h[j, :j].dot(current[:j]) + h[j, j + 1:].dot(prev[j + 1:]) - prev[j] + g[j])
        result.append(current)

    return result

a = np.array([[9.331343,1.120045,-2.880925],[1.120045,7.086042,0.670297],[-2.880925,0.670297,5.622534]], float)
b = np.array([7.570463,8.876384,3.411906], float)

solution = np.linalg.solve(a, b)

n, _ = a.shape
d = np.identity(n, float) * a

hd = np.identity(n, float) - np.linalg.inv(d).dot(a)
hdNorm = np.linalg.norm(hd, np.inf)
gd = np.linalg.inv(d).dot(b)
x0 = np.zeros(n, float)
k = 7

xs = iterations(hd, gd, x0, k)
wa, _ = np.linalg.eig(hd)
wa = np.sort(np.abs(wa))
#print(wa)

print("Accurate solution (x*):")
print(solution)
print("|| H_D || = ", np.linalg.norm(hd, np.inf))
print()

print("Solution by simple iteration method:")
print(xs[k])
print("|| x(7) - x* ||  = ", np.linalg.norm(solution - xs[k], np.inf))
print("|| x(7) - x* || <= ", priorEstimate(hd, gd, x0, k), "(prior estimate)")
print("|| x(7) - x* || <= ", hdNorm / (1. - hdNorm) * np.linalg.norm(xs[k] - xs[k - 1], np.inf), "(posterior estimate)")
print("Solution by Lyusternik's method:")
lyusternik = xs[k - 1] + (xs[k] - xs[k - 1]) / (1. - wa[n - 1])
print(lyusternik)
print("|| x(7)L - x* || = ", np.linalg.norm(solution - lyusternik, np.inf))
print()

print("Solution by Seidel's method:")
seidels = seidel(hd, gd, x0, k)
print(seidels[k])
print("|| x(7)S - x* || = ", np.linalg.norm(solution - seidels[k], np.inf))
hl = np.tril(hd, -1)
hr = np.triu(hd)
print("pho(H_D) =", spectralRadius(np.linalg.inv(np.identity(n, float) - hl).dot(hr)))
print()

print("Solution by upper relaxation:")
relaxations = upperRelaxation(hd, gd, x0, k)
print(relaxations[k])
print("|| x(7)R - x* || = ", np.linalg.norm(solution - relaxations[k], np.inf))