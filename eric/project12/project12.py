#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import numpy as np

from ..utils import print_mat
from ..utils import np_load


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--stub', default="h2o_sto3g")
    parser.add_argument('--nbasis', type=int, default=7)
    parser.add_argument('--nelec', type=int, default=10)
    parser.add_argument('--thresh-e', type=int, default=15)

    args = parser.parse_args()

    return args


def mo_so_4index(TEI_SO, TEI_MO):

    norb = TEI_MO.shape[0]

    for p in range(2 * norb):
        for q in range(2 * norb):
            for r in range(2 * norb):
                for s in range(2 * norb):
                    lint = TEI_MO[p//2, r//2, q//2, s//2] * (p % 2 == r % 2) * (q % 2 == s % 2)
                    rint = TEI_MO[p//2, s//2, q//2, r//2] * (p % 2 == s % 2) * (q % 2 == r % 2)
                    TEI_SO[p, q, r, s] = lint - rint

    return


def build_fock_spin_orbital(TEI_SO, H, nsocc):

    nsorb = TEI_SO.shape[0]

    F_SO = np.zeros(shape=(nsorb, nsorb))

    for p in range(nsorb):
        for q in range(nsorb):
            F_SO[p, q] += H[p//2, q//2]
            for m in range(nsocc):
                F_SO[p, q] += TEI_SO[p, m, q, m]

    return F_SO


def make_spin_orbital_energies(E_MO):
    N_MO = E_MO.shape[0]
    N_SO = N_MO * 2
    E_SO = np.zeros((N_SO))
    for p in range(N_SO):
        E_SO[p] = E_MO[p//2, p//2]
    return np.diag(E_SO)


def form_hamiltonian_cis(H_CIS, E_SO, TEI_SO, nsocc):
    """Form the CIS Hamiltonian in the spin orbital basis."""

    # nsov = H_CIS.shape[0]
    nsorb = E_SO.shape[0]
    nsvir = nsorb - nsocc

    for i in range(nsocc):
        for a in range(nsvir):
            ia = i*nsvir + a
            for j in range(nsocc):
                for b in range(nsvir):
                    jb = j*nsvir + b
                    # print('i: {} a: {} ia: {} j: {} b: {} jb: {}'.format(i, a, ia, j, b, jb))
                    H_CIS[ia, jb] = ((i == j) * E_SO[a + nsocc, b + nsocc]) - ((a == b) * E_SO[i, j]) + TEI_SO[a + nsocc, j, i, b + nsocc]

    return


def form_hamiltonian_cis_singlet(H_CIS_singlet, E_MO, TEI_MO, nocc):
    """Form the singlet CIS Hamiltonian in the molecular orbital basis."""

    norb = E_MO.shape[0]
    nvirt = norb - nocc

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    # 2<aj|ib> - <aj|bi> = 2(ai|jb) - (ab|ji)
                    H_CIS_singlet[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) + (2 * TEI_MO[a + nocc, i, j, b + nocc]) - TEI_MO[a + nocc, b + nocc, j, i]

    return


def form_hamiltonian_cis_triplet(H_CIS_triplet, E_MO, TEI_MO, nocc):
    """Form the triplet CIS Hamiltonian in the molecular orbital basis."""

    norb = E_MO.shape[0]
    nvirt = norb - nocc

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    # <aj|bi> = (ab|ji)
                    H_CIS_triplet[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) - TEI_MO[a + nocc, b + nocc, j, i]

    return


# def form_rpa_a_matrix(E_MO, TEI_MO):
#     """Form the A (CIS) matrix for RPA."""

#     norb = TEI_MO.shape[0]
#     nvirt = norb - nocc
#     nov = nocc * nvirt

#     A = np.empty(shape=(nov, nov))

#     for i in range(nocc):
#         for a in range(nvirt):
#             ia = i*nvirt + a
#             for j in range(nocc):
#                 for b in range(nvirt):
#                     jb = j*nvirt + b
#                     # <aj||ib> = <aj|ib> - <aj|bi> = (ai|jb) - (ab|ji)
#                     A[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) + TEI_MO[a + nocc, i, j, b + nocc] - TEI_MO[a + nocc, b + nocc, j, i]

#     return A


# def form_rpa_b_matrix(TEI_MO):
#     """Form the B matrix for RPA."""

#     norb = TEI_MO.shape[0]
#     nvirt = norb - nocc
#     nov = nocc * nvirt

#     B = np.empty(shape=(nov, nov))

#     for i in range(nocc):
#         for a in range(nvirt):
#             ia = i*nvirt + a
#             for j in range(nocc):
#                 for b in range(nvirt):
#                     jb = j*nvirt + b
#                     # <ab||ij> = <ab|ij> - <ab|ji> = (ai|bj) - (aj|bi)
#                     B[ia, jb] = TEI_MO[a + nocc, i, b + nocc, j] - TEI_MO[a + nocc, j, b + nocc, i]

#     return B


def form_rpa_a_matrix_so(E_SO, TEI_SO, nocc):
    """Form the A (CIS) matrix for RPA in the spin orbital basis."""

    norb = TEI_SO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = ((i == j) * E_SO[a + nocc, b + nocc]) - ((a == b) * E_SO[i, j]) + TEI_SO[a + nocc, j, i, b + nocc]

    return A


def form_rpa_b_matrix_so(TEI_SO, nocc):
    """Form the B matrix for RPA in the spin orbital basis."""

    norb = TEI_SO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = TEI_SO[a + nocc, b + nocc, i, j]

    return B


if __name__ == "__main__":

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    norb = dim = nbasis = args.nbasis
    nvirt = norb - nocc
    nov = nocc * nvirt

    E_MO = np_load('F_MO.npz')
    H = np_load('H.npz')
    TEI_MO = np_load('TEI_MO.npz')
    print('MO energies')
    print(np.diag(E_MO))
    # sum_all = np.sum(E_MO)
    # sum_diag = np.sum(np.diag(E_MO))
    # diff = sum_all - sum_diag
    # print(sum_all, sum_diag, diff)
    TEI_SO = np.zeros(shape=np.array(TEI_MO.shape) * 2)
    E_SO = make_spin_orbital_energies(E_MO)
    nsorb = TEI_SO.shape[0]
    nsocc = nocc * 2
    nsvir = nsorb - nsocc
    nsov = nsocc * nsvir
    print('nsorb: {} nsocc: {} nsvir: {} nsov: {}'.format(nsorb, nsocc, nsvir, nsov))
    # Transform the two-electron integrals from the spatial MO-basis
    # to the spin-orbital basis.
    mo_so_4index(TEI_SO, TEI_MO)
    # Build the Fock matrix in the SO basis.
    F_SO = build_fock_spin_orbital(TEI_SO, H, nsocc)

    H_CIS = np.zeros(shape=(nsov, nsov))
    # Form the CIS Hamiltonian.
    form_hamiltonian_cis(H_CIS, E_SO, TEI_SO, nsocc)
    energies_CIS, eigvals_CIS = np.linalg.eigh(H_CIS)
    print('CIS excitation energies (SO basis)')
    hartree_to_ev = 27.211385
    for i, e in enumerate(energies_CIS, start=1):
        print(i, e, e * hartree_to_ev)

    ## Spin-adapted CIS
    H_CIS_singlet = np.zeros(shape=(nov, nov))
    H_CIS_triplet = np.zeros(shape=(nov, nov))
    form_hamiltonian_cis_singlet(H_CIS_singlet, E_MO, TEI_MO, nocc)
    form_hamiltonian_cis_triplet(H_CIS_triplet, E_MO, TEI_MO, nocc)
    energies_CIS_singlet, eigvecs_CIS_singlet = np.linalg.eigh(H_CIS_singlet)
    energies_CIS_triplet, eigvecs_CIS_triplet = np.linalg.eigh(H_CIS_triplet)
    _energies_CIS_singlet = sorted(((e, 'singlet') for e in energies_CIS_singlet))
    _energies_CIS_triplet = sorted(((e, 'triplet') for e in energies_CIS_triplet))
    energies_CIS = sorted(_energies_CIS_singlet + _energies_CIS_triplet)
    print('CIS excitation energies (MO basis)')
    for i, (e, t) in enumerate(energies_CIS, start=1):
        print(i, e, e * hartree_to_ev, t)

    ## Time-Dependent Hartree-Fock (TDHF) / Random Phase Approximation (RPA)
    A = form_rpa_a_matrix_so(E_SO, TEI_SO, nsocc)
    B = form_rpa_b_matrix_so(TEI_SO, nsocc)
    # Form the RPA supermatrix.
    H_RPA = np.bmat([[ A,  B],
                     [-B, -A]])
    eigvals_RPA, eigvecs_RPA = np.linalg.eig(H_RPA)
    idx_RPA = eigvals_RPA.argsort()
    eigvals_RPA = eigvals_RPA[idx_RPA].real
    eigvecs_RPA = eigvecs_RPA[idx_RPA].real
    print('RPA excitation energies (SO basis), method 1')
    for i, e in enumerate(eigvals_RPA, start=1):
        print(i, e, e * hartree_to_ev)

    H_RPA_reduced = np.dot(A + B, A - B)
    eigvals_RPA_reduced, eigvecs_RPA_reduced = np.linalg.eig(H_RPA_reduced)
    idx_RPA_reduced = eigvals_RPA_reduced.argsort()
    eigvals_RPA_reduced = np.sqrt(eigvals_RPA_reduced[idx_RPA_reduced].real)
    eigvecs_RPA_reduced = eigvecs_RPA_reduced[idx_RPA_reduced].real
    print('RPA excitation energies (SO basis), method 2')
    for i, e in enumerate(eigvals_RPA_reduced, start=1):
        print(i, e, e * hartree_to_ev)
