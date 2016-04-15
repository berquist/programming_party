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
    """Form the CIS Hamiltonian."""

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


if __name__ == "__main__":

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    norb = dim = nbasis = args.nbasis
    nvirt = norb - nocc
    nov = nocc * nvirt

    E = np_load('F_MO.npz')
    H = np_load('H.npz')
    TEI_MO = np_load('TEI_MO.npz')
    print('MO energies')
    print(np.diag(E))
    print('E.shape: {}'.format(E.shape))
    print('H.shape: {}'.format(H.shape))
    print('TEI_MO.shape: {}'.format(TEI_MO.shape))
    # sum_all = np.sum(E)
    # sum_diag = np.sum(np.diag(E))
    # diff = sum_all - sum_diag
    # print(sum_all, sum_diag, diff)
    TEI_SO = np.zeros(shape=np.array(TEI_MO.shape) * 2)
    print('TEI_SO.shape: {}'.format(TEI_SO.shape))
    E_SO = make_spin_orbital_energies(E)
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
    # print_mat(H_CIS)
    # print_mat(H_CIS[-7:, -7:])
