#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np

from ..utils import np_load


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--nelec', type=int, default=10)

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
    print('sum(ints): {}'.format(np.sum(TEI_SO)))
    return


def make_amplitude_guess(TEI_SO, E, nsocc):

    nsorb = TEI_SO.shape[0]

    T1 = np.zeros(shape=(nsorb, nsorb))
    T2 = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    T2[i, j, a, b] += TEI_SO[i, j, a, b] / (E[i//2] + E[j//2] - E[a//2] - E[b//2])

    return T1, T2


def calc_mp2_energy(TEI_SO, T2, nsocc):

    nsorb = TEI_SO.shape[0]

    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)

    E_MP2 = np.einsum('ijab,ijab->', TEI_SO[o, o, v, v], T2[o, o, v, v]) / 4.0

    return E_MP2


def build_fock_spin_orbital(TEI_SO, H, nsocc):

    nsorb = TEI_SO.shape[0]

    F_SO = np.zeros(shape=(nsorb, nsorb))

    for p in range(nsorb):
        for q in range(nsorb):
            F_SO[p, q] += H[p//2, q//2]
            for m in range(nsocc):
                F_SO[p, q] += TEI_SO[p, m, q, m]

    return F_SO


def build_diagonal_2(F_SO, nsocc):
    """Build the 2-index energy matrix."""

    nsorb = F_SO.shape[0]

    D = np.zeros(shape=(nsorb, nsorb))

    for i in range(0, nsocc):
        for a in range(nsocc, nsorb):
            D[i, a] += (F_SO[i, i] - F_SO[a, a])
    print('sum(D2): {}'.format(np.sum(D)))
    return D


def build_diagonal_4(F_SO, nsocc):
    """Build the 4-index energy matrix."""

    nsorb = F_SO.shape[0]

    D = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    D[i, j, a, b] += (F_SO[i, i] + F_SO[j, j] - F_SO[a, a] - F_SO[b, b])
    print('sum(D4): {}'.format(np.sum(D)))
    return D


def build_intermediates_2index(TEI_SO, F_SO, T1, T2, tau_twiddle, nsocc):
    """Form the 2-index intermediate F using equations 3-5."""

    nsorb = TEI_SO.shape[0]

    F = np.zeros(shape=(nsorb, nsorb))

    # equation (3)
    for a in range(nsocc, nsorb):
        for e in range(nsocc, nsorb):
            F[a, e] += (1 - int(a == e)) * F_SO[a, e]
            for m in range(0, nsocc):
                F[a, e] += (-0.5) * F_SO[m, e] * T1[m, a]
                for f in range(nsocc, nsorb):
                    F[a, e] += T1[m, f] * TEI_SO[m, a, f, e]
                    for n in range(0, nsocc):
                        F[a, e] += (-0.5) * tau_twiddle[m, n, a, f] * TEI_SO[m, n, e, f]

    # equation (4)
    for m in range(0, nsocc):
        for i in range(0, nsocc):
            F[m, i] += (1 - int(m == i)) * F_SO[m, i]
            for e in range(nsocc, nsorb):
                F[m, i] += 0.5 * T1[i, e] * F_SO[m, e]
                for n in range(0, nsocc):
                    F[m, i] += T1[n, e] * TEI_SO[m, n, i, e]
                    for f in range(nsocc, nsorb):
                        F[m, i] += 0.5 * tau_twiddle[i, n, e, f] * TEI_SO[m, n, e, f]

    # equation (5)
    for m in range(0, nsocc):
        for e in range(nsocc, nsorb):
            F[m, e] += F_SO[m, e]
            for n in range(0, nsocc):
                for f in range(nsocc, nsorb):
                    F[m, e] += T1[n, f] * TEI_SO[m, n, e, f]

    return F


def form_tau_and_twiddle(tau, tau_twiddle, T1, T2, nsocc):
    """Form the effective two-particle excitation operators tau and tau
    twiddle.
    """

    nsorb = T1.shape[0]

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    singles = (T1[i, a] * T1[j, b]) - (T1[i, b] * T1[j, a])
                    tau_twiddle[i, j, a, b] = T2[i, j, a, b] + 0.5 * singles
                    tau[i, j, a, b] = T2[i, j, a, b] + singles

    return


def build_intermediates_4index(TEI_SO, F_SO, T1, T2, tau, nsocc):
    """Form the 4-index intermediate W using equations 6-8."""

    nsorb = TEI_SO.shape[0]

    W = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    # equation (6)
    for m in range(0, nsocc):
        for n in range(0, nsocc):
            for i in range(0, nsocc):
                for j in range(0, nsocc):
                    W[m, n, i, j] += TEI_SO[m, n, i, j]
                    for e in range(nsocc, nsorb):
                        normal = T1[j, e] * TEI_SO[m, n, i, e]
                        perm_ij = T1[i, e] * TEI_SO[m, n, j, e]
                        W[m, n, i, j] += (normal - perm_ij)
                        for f in range(nsocc, nsorb):
                            W[m, n, i, j] += 0.25 * tau[i, j, e, f] * TEI_SO[m, n, e, f]

    # equation (7)
    for a in range(nsocc, nsorb):
        for b in range(nsocc, nsorb):
            for e in range(nsocc, nsorb):
                for f in range(nsocc, nsorb):
                    W[a, b, e, f] += TEI_SO[a, b, e, f]
                    for m in range(0, nsocc):
                        normal = T1[m, b] * TEI_SO[a, m, e, f]
                        perm_ab = T1[m, a] * TEI_SO[b, m, e, f]
                        W[a, b, e, f] -= (normal - perm_ab)
                        for n in range(0, nsocc):
                            W[a, b, e, f] += (0.25) * tau[m, n, a, b] * TEI_SO[m, n, e, f]

    # equation (8)
    for m in range(0, nsocc):
        for b in range(nsocc, nsorb):
            for e in range(nsocc, nsorb):
                for j in range(0, nsocc):
                    W[m, b, e, j] += TEI_SO[m, b, e, j]
                    for f in range(nsocc, nsorb):
                        W[m, b, e, j] += T1[j, f] * TEI_SO[m, b, e, f]
                    for n in range(0, nsocc):
                        W[m, b, e, j] -= T1[n, b] * TEI_SO[m, n, e, j]
                        for f in range(nsocc, nsorb):
                            W[m, b, e, j] -= (0.5*T2[j, n, f, b] + T1[j, f]*T1[n, b]) * TEI_SO[m, n, e, f]

    return W


def update_amplitudes_T1(TEI_SO, F_SO, T1, T2, nsocc, D1, F):
    """Update the T1 amplitudes."""

    nsorb = F_SO.shape[0]

    uT1 = np.zeros(shape=(nsorb, nsorb))

    for i in range(0, nsocc):
        for a in range(nsocc, nsorb):
            uT1[i, a] += F_SO[i, a]
            for e in range(nsocc, nsorb):
                uT1[i, a] += T1[i, e] * F[a, e]
            for m in range(0, nsocc):
                uT1[i, a] -= T1[m, a] * F[m, i]
                for e in range(nsocc, nsorb):
                    uT1[i, a] += T2[i, m, a, e] * F[m, e]
            for n in range(0, nsocc):
                for f in range(nsocc, nsorb):
                    uT1[i, a] -= T1[n, f] * TEI_SO[n, a, i, f]
            for m in range(0, nsocc):
                for e in range(nsocc, nsorb):
                    for f in range(nsocc, nsorb):
                        uT1[i, a] -= 0.5 * T2[i, m, e, f] * TEI_SO[m, a, e, f]
                    for n in range(0, nsocc):
                        uT1[i, a] -= 0.5 * T2[m, n, a, e] * TEI_SO[n, m, e, i]
            uT1[i, a] /= D1[i, a]

    return uT1


def update_amplitudes_T2(TEI_SO, F_SO, T1, T2, nsocc, D2, W, tau):
    """Update the T2 amplitudes."""

    nsorb = F_SO.shape[0]

    uT2 = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    # 0
                    uT2[i, j, a, b] += TEI_SO[i, j, a, b]
                    # 1
                    normal = 0.0
                    perm_ab = 0.0
                    for e in range(nsocc, nsorb):
                        tmp_normal = 0.0
                        tmp_perm_ab = 0.0
                        for m in range(0, nsocc):
                            tmp_normal += (-0.5) * T1[m, b] * F_SO[m, e]
                            tmp_perm_ab += (-0.5) * T1[m, a] * F_SO[m, e]
                        normal += T2[i, j, a, e] * (F_SO[b, e] + tmp_normal)
                        perm_ab += T2[i, j, b, e] * (F_SO[a, e] + tmp_perm_ab)
                    uT2[i, j, a, b] += (normal - perm_ab)
                    # 2
                    normal = 0.0
                    perm_ij = 0.0
                    for m in range(0, nsocc):
                        tmp_normal = 0.0
                        tmp_perm_ij = 0.0
                        for e in range(nsocc, nsorb):
                            tmp_normal += (0.5) * T1[j, e] * F_SO[m, e]
                            tmp_perm_ij += (0.5) * T1[i, e] * F_SO[m, e]
                        normal += (-1.0) * T2[i, m, a, b] * (F_SO[m, j] + tmp_normal)
                        perm_ij += (-1.0) * T2[j, m, a, b] * (F_SO[m, i] + tmp_perm_ij)
                    uT2[i, j, a, b] += (-1.0) * (normal - perm_ij)
                    # 3
                    for m in range(0, nsocc):
                        for n in range(0, nsocc):
                            uT2[i, j, a, b] += (0.5) * tau[m, n, a, b] * W[m, n, i, j]
                    for e in range(nsocc, nsorb):
                        for f in range(nsocc, nsorb):
                            uT2[i, j, a, b] += (0.5) * tau[i, j, e, f] * W[a, b, e, f]
                    # 4
                    normal = 0.0
                    perm_ij = 0.0
                    perm_ab = 0.0
                    perm_ij_ab = 0.0
                    for m in range(0, nsocc):
                        for e in range(nsocc, nsorb):
                            normal  += (T2[i, m, a, e] * W[m, b, e, j]) - (T1[i, e] * T1[m, a] * TEI_SO[m, b, e, j])
                            perm_ij += (T2[j, m, a, e] * W[m, b, e, i]) - (T1[j, e] * T1[m, a] * TEI_SO[m, b, e, i])
                            perm_ab += (T2[i, m, b, e] * W[m, a, e, j]) - (T1[i, e] * T1[m, b] * TEI_SO[m, a, e, j])
                            perm_ij_ab += (T2[j, m, b, e] * W[m, a, e, i]) - (T1[j, e] * T1[m, b] * TEI_SO[m, a, e, i])
                    uT2[i, j, a, b] += (normal - perm_ij - perm_ab + perm_ij_ab)
                    # 5
                    normal = 0.0
                    perm_ij = 0.0
                    for e in range(nsocc, nsorb):
                        normal += T1[i, e] * TEI_SO[a, b, e, j]
                        perm_ij += T1[j, e] * TEI_SO[a, b, e, i]
                    uT2[i, j, a, b] += (normal - perm_ij)
                    normal = 0.0
                    perm_ab = 0.0
                    for m in range(0, nsocc):
                        normal += T1[m, a] * TEI_SO[m, b, i, j]
                        perm_ab += T1[m, b] * TEI_SO[m, a, i, j]
                    uT2[i, j, a, b] -= (normal - perm_ab)
                    uT2[i, j, a, b] /= D2[i, j, a, b]

    return uT2


def calc_ccsd_energy(TEI_SO, F_SO, T1, T2, nsocc):

    nsorb = TEI_SO.shape[0]

    E_CCSD = 0.0

    for i in range(0, nsocc):
        for a in range(nsocc, nsorb):
            E_CCSD += F_SO[i, a] * T1[i, a]

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    E_CCSD += 0.25 * TEI_SO[i, j, a, b] * T2[i, j, a, b]
                    E_CCSD += 0.50 * TEI_SO[i, j, a, b] * T1[i, a] * T1[j, b]

    return E_CCSD


def main():

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    nsocc = nocc * 2

    H = np_load('H.npz')
    E = np.diag(np_load('F_MO.npz'))
    TEI_MO = np_load('TEI_MO.npz')

    TEI_SO = np.zeros(shape=np.array(TEI_MO.shape) * 2)
    nsorb = TEI_SO.shape[0]
    # Transform the two-electron integrals from the spatial MO-basis
    # to the spin-orbital basis.
    mo_so_4index(TEI_SO, TEI_MO)

    # Build the Fock matrix in the SO basis.
    F_SO = build_fock_spin_orbital(TEI_SO, H, nsocc)
    # Use the SO Fock matrix to build the diagonal energy arrays.
    D1 = build_diagonal_2(F_SO, nsocc)
    D2 = build_diagonal_4(F_SO, nsocc)

    # Make an initial guess at the T1 and T2 cluster amplitudes.
    T1, T2 = make_amplitude_guess(TEI_SO, E, nsocc)

    # If everything's working right up to this point, we should be
    # able to calculate the MP2 energy.
    E_MP2 = calc_mp2_energy(TEI_SO, T2, nsocc)

    assert abs(E_MP2 - -0.049149636120) < 1.0e-14

    print('E(MP2): {:20.12f}'.format(E_MP2))

    # Form the effective two-particle excitation operators.
    tau = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))
    tau_twiddle = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))
    form_tau_and_twiddle(tau, tau_twiddle, T1, T2, nsocc)

    # Form the one- and two-particle intermediates.
    F = build_intermediates_2index(TEI_SO, F_SO, T1, T2, tau_twiddle, nsocc)
    W = build_intermediates_4index(TEI_SO, F_SO, T1, T2, tau, nsocc)

    # Calculate the first updated set of amplitudes.
    T1 = update_amplitudes_T1(TEI_SO, F_SO, T1, T2, nsocc, D1, F)
    T2 = update_amplitudes_T2(TEI_SO, F_SO, T1, T2, nsocc, D2, W, tau)

    E_CCSD = calc_ccsd_energy(TEI_SO, F_SO, T1, T2, nsocc)

    print('Iter E(CCSD) sum(T1) sum(T2)')
    t = ' {:4d} {:20.12f} {} {}'.format
    print(t(0, E_CCSD, np.sum(T1), np.sum(T2)))

    thresh_e = 1.0e-15
    iteration = 1
    max_iterations = 1024

    while iteration < max_iterations:

        form_tau_and_twiddle(tau, tau_twiddle, T1, T2, nsocc)

        F = build_intermediates_2index(TEI_SO, F_SO, T1, T2, tau_twiddle, nsocc)
        W = build_intermediates_4index(TEI_SO, F_SO, T1, T2, tau, nsocc)

        T1 = update_amplitudes_T1(TEI_SO, F_SO, T1, T2, nsocc, D1, F)
        T2 = update_amplitudes_T2(TEI_SO, F_SO, T1, T2, nsocc, D2, W, tau)

        E_CCSD = calc_ccsd_energy(TEI_SO, F_SO, T1, T2, nsocc)

        print(t(iteration, E_CCSD, np.sum(T1), np.sum(T2)))

        iteration += 1

    return


if __name__ == '__main__':
    main()
