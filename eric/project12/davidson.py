"""A block Davidson solver for finding a fixed number of eigenvalues.

Adapted from https://joshuagoings.com/2013/08/23/davidsons-method/
"""
import time

import numpy as np

# dimension of problem
n = 1200
# convergence tolerance
tol = 1e-8
# maximum number of iterations
mmax = n // 2

## set up fake Hamiltonian

sparsity = 1.0e-4
A = np.zeros((n, n))
for i in range(0, n):
    A[i, i] = i + 1
A = A + sparsity * np.random.randn(n, n)
A = (A.T + A) / 2

## set up subspace and trial vectors

# number of initial guess vectors
k = 8
# number of eigenvalues to solve
eig = 4
# set of k unit vectors as guess
t = np.eye(n, k)
# hold guess vectors
V = np.zeros((n, n))
I = np.eye(n)

# Begin block Davidson routine

start_davidson = time.time()

for m in range(k, mmax, k):
    if m <= k:
        for j in range(k):
            V[:, j] = t[:, j] / np.linalg.norm(t[:, j])
        theta_old = 1
    elif m > k:
        theta_old = theta[:eig]
    V, R = np.linalg.qr(V)
    T = V[:, : (m + 1)].T @ A @ V[:, : (m + 1)]
    THETA, S = np.linalg.eig(T)
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:, idx]
    for j in range(k):
        w = (A - theta[j] * I) @ V[:, : (m + 1)] @ s[:, j]
        q = w / (theta[j] - A[j, j])
        V[:, (m + j + 1)] = q
    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break

end_davidson = time.time()

# End of block Davidson. Print results.

print(f"davidson = {theta[:eig]}; {end_davidson - start_davidson} seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E, Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print(f"numpy = {E[:eig]}; {end_numpy - start_numpy} seconds")
