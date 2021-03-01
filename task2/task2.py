import numpy as np

a = np.array([[9.331343,1.120045,-2.880925],[1.120045,7.086042,0.670297],[-2.880925,0.670297,5.622534]], float)
b = np.array([7.570463,8.876384,3.411906], float)

solution = np.linalg.solve(a, b)

n, _ = a.shape
d = np.identity(n, float) * a

hd = np.identity(n, float) - np.linalg.inv(d).dot(a)
gd = np.linalg.inv(d).dot(b)

print("|| H_D ||_infinity = ", np.linalg.norm(hd, np.inf))