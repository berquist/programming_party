"""A block Davidson solver for finding a fixed number of eigenvalues.

Adapted from https://joshuagoings.com/2013/08/23/davidsons-method/
"""
import time
from typing import Tuple

import numpy as np
from tqdm import tqdm


def davidson(A: np.ndarray, k: int, eig: int) -> Tuple[np.ndarray, np.ndarray]:
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    ## set up subspace and trial vectors

    # set of k unit vectors as guess
    t = np.eye(n, k)
    # hold guess vectors
    V = np.zeros((n, n))
    I = np.eye(n)

    for m in tqdm(range(k, mmax, k)):
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
    return theta, V


if __name__ == "__main__":

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

    # number of initial guess vectors
    k = 8
    # number of eigenvalues to solve
    eig = 4

    start_davidson = time.time()
    theta, V = davidson(A, k, eig)
    end_davidson = time.time()

    print(f"davidson = {theta[:eig]}; {end_davidson - start_davidson} seconds")

    start_numpy = time.time()
    E, Vec = np.linalg.eig(A)
    E = np.sort(E)
    end_numpy = time.time()

    print(f"numpy = {E[:eig]}; {end_numpy - start_numpy} seconds")
