#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import numpy as np
np_formatter = {
    'float_kind': lambda x: '{:14.8f}'.format(x)
}
np.set_printoptions(linewidth=160, formatter=np_formatter, threshold=np.inf)
import scipy.constants as spc

from ..utils import matsym
from ..utils import np_load
from ..utils import print_mat
from ..utils import read_arma_mat_ascii

from ..project3.project3 import parse_int_file_2


hartree_to_ev = spc.physical_constants['Hartree energy in eV'][0]


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
    """Place the molecular orbital energies in a diagonal matrix indexed
    identially to the spin-orbital basis quantities.
    """

    N_MO = E_MO.shape[0]
    N_SO = N_MO * 2
    E_SO = np.zeros((N_SO))

    for p in range(N_SO):
        E_SO[p] = E_MO[p//2, p//2]

    return np.diag(E_SO)


def form_hamiltonian_cis_so(H_CIS, E_SO, TEI_SO, nsocc):
    """Form the CIS Hamiltonian in the spin orbital (SO) basis.

    The equation for element {ia,jb} is <aj||ib> = <aj|ib> - <aj|bi> =
    [ai|jb] - [ab|ji]. It also includes the virt-occ energy difference
    on the diagonal.
    """

    nsorb = E_SO.shape[0]
    nsvir = nsorb - nsocc

    for i in range(nsocc):
        for a in range(nsvir):
            ia = i*nsvir + a
            for j in range(nsocc):
                for b in range(nsvir):
                    jb = j*nsvir + b
                    H_CIS[ia, jb] = ((i == j) * E_SO[a + nsocc, b + nsocc]) - ((a == b) * E_SO[i, j]) + TEI_SO[a + nsocc, j, i, b + nsocc]

    return


def form_hamiltonian_cis_mo_singlet(H_CIS_singlet, E_MO, TEI_MO, nocc):
    """Form the singlet CIS Hamiltonian in the molecular orbital (MO) basis.

    The equation for element {ia,jb} is <aj||ib> = <aj|ib> - <aj|bi> =
    [ai|jb] - [ab|ji] = 2(ai|jb) - (ab|ji). It also includes the
    virt-occ energy difference on the diagonal.
    """

    norb = E_MO.shape[0]
    nvirt = norb - nocc

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    H_CIS_singlet[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) + (2 * TEI_MO[a + nocc, i, j, b + nocc]) - TEI_MO[a + nocc, b + nocc, j, i]

    return


def form_hamiltonian_cis_mo_triplet(H_CIS_triplet, E_MO, TEI_MO, nocc):
    """Form the triplet CIS Hamiltonian in the molecular orbital (MO) basis.

    The equation for element {ia,jb} is -<aj|bi> = -(ab|ji). It also
    includes the virt-occ energy difference on the diagonal.
    """

    norb = E_MO.shape[0]
    nvirt = norb - nocc

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    H_CIS_triplet[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) - TEI_MO[a + nocc, b + nocc, j, i]

    return


def form_rpa_a_matrix_so(E_SO, TEI_SO, nsocc):
    """Form the A (CIS) matrix for RPA in the spin orbital (SO) basis.

    The equation for element {ia,jb} is <aj||ib> = <aj|ib> - <aj|bi> =
    [ai|jb] - [ab|ji]. It also includes the virt-occ energy difference
    on the diagonal.
    """

    nsorb = TEI_SO.shape[0]
    nsvir = nsorb - nsocc
    nov = nsocc * nsvir

    A = np.empty(shape=(nov, nov))

    for i in range(nsocc):
        for a in range(nsvir):
            ia = i*nsvir + a
            for j in range(nsocc):
                for b in range(nsvir):
                    jb = j*nsvir + b
                    A[ia, jb] = ((i == j) * E_SO[a + nsocc, b + nsocc]) - ((a == b) * E_SO[i, j]) + TEI_SO[a + nsocc, j, i, b + nsocc]

    return A


def form_rpa_a_matrix_mo_singlet(E_MO, TEI_MO, nocc):
    """Form the A (CIS) matrix for RPA in the molecular orbital (MO)
    basis. [singlet]

    The equation for element {ia,jb} is <aj||ib> = <aj|ib> - <aj|bi> =
    [ai|jb] - [ab|ji] = 2(ai|jb) - (ab|ji). It also includes the
    virt-occ energy difference on the diagonal.
    """

    norb = E_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = 2*TEI_MO[a + nocc, i, j, b + nocc] - TEI_MO[a + nocc, b + nocc, j, i]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A

def form_rpa_a_matrix_mo_triplet(E_MO, TEI_MO, nocc):
    """Form the A (CIS) matrix for RPA in the molecular orbital (MO)
    basis. [triplet]

    The equation for element {ia,jb} is - <aj|bi> = - [ab|ji] = -
    (ab|ji). It also includes the virt-occ energy difference on the
    diagonal.
    """

    norb = E_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = - TEI_MO[a + nocc, b + nocc, j, i]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A

def form_rpa_b_matrix_so(TEI_SO, nocc):
    """Form the B matrix for RPA in the spin orbital (SO) basis.

    The equation for element {ia,jb} is <ab||ij> = <ab|ij> - <ab|ji> =
    [ai|bj] - [aj|bi].
    """

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


def form_rpa_b_matrix_mo_singlet(TEI_MO, nocc):
    """Form the B matrix for RPA in the molecular orbital (MO)
    basis. [singlet]

    The equation for element {ia,jb} is <ab||ij> = <ab|ij> - <ab|ji> =
    [ai|bj] - [aj|bi] = 2*(ai|bj) - (aj|bi).
    """

    norb = TEI_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = 2*TEI_MO[a + nocc, i, b + nocc, j] - TEI_MO[a + nocc, j, b + nocc, i]

    return B


def form_rpa_b_matrix_mo_triplet(TEI_MO, nocc):
    """Form the B matrix for RPA in the molecular orbital (MO)
    basis. [triplet]

    The equation for element {ia,jb} is <ab||ij> = <ab|ij> - <ab|ji> =
    [ai|bj] - [aj|bi] = 2(ai|bj) - (aj|bi).
    """

    norb = TEI_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = 2*TEI_MO[a + nocc, i, b + nocc, j] - TEI_MO[a + nocc, j, b + nocc, i]

    return B


if __name__ == "__main__":

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    norb = dim = nbasis = args.nbasis
    nvirt = norb - nocc
    nov = nocc * nvirt

    C = np_load('C.npz')
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

    H_CIS_SO = np.zeros(shape=(nsov, nsov))
    form_hamiltonian_cis_so(H_CIS_SO, E_SO, TEI_SO, nsocc)
    assert matsym(H_CIS_SO) == 1
    energies_CIS_SO, eigvecs_CIS_SO = np.linalg.eigh(H_CIS_SO)
    idx_CIS_SO = energies_CIS_SO.argsort()
    energies_CIS_SO = energies_CIS_SO[idx_CIS_SO]
    eigvecs_CIS_SO = eigvecs_CIS_SO[:, idx_CIS_SO]
    print('CIS excitation energies (SO basis)')
    for i, e in enumerate(energies_CIS_SO, start=1):
        print(i, e, e * hartree_to_ev)

    ## Spin-adapted CIS
    H_CIS_MO_singlet = np.zeros(shape=(nov, nov))
    H_CIS_MO_triplet = np.zeros(shape=(nov, nov))
    form_hamiltonian_cis_mo_singlet(H_CIS_MO_singlet, E_MO, TEI_MO, nocc)
    form_hamiltonian_cis_mo_triplet(H_CIS_MO_triplet, E_MO, TEI_MO, nocc)
    assert matsym(H_CIS_MO_singlet) == 1
    assert matsym(H_CIS_MO_triplet) == 1
    energies_CIS_MO_singlet, eigvecs_CIS_MO_singlet = np.linalg.eigh(H_CIS_MO_singlet)
    energies_CIS_MO_triplet, eigvecs_CIS_MO_triplet = np.linalg.eigh(H_CIS_MO_triplet)
    idx_CIS_MO_singlet = energies_CIS_MO_singlet.argsort()
    idx_CIS_MO_triplet = energies_CIS_MO_triplet.argsort()
    energies_CIS_MO_singlet = energies_CIS_MO_singlet[idx_CIS_MO_singlet]
    energies_CIS_MO_triplet = energies_CIS_MO_triplet[idx_CIS_MO_triplet]
    eigvecs_CIS_MO_singlet = eigvecs_CIS_MO_singlet[:, idx_CIS_MO_singlet]
    eigvecs_CIS_MO_triplet = eigvecs_CIS_MO_triplet[:, idx_CIS_MO_triplet]
    energies_CIS_MO_singlet_labeled = sorted(((e, 'singlet') for e in energies_CIS_MO_singlet))
    energies_CIS_MO_triplet_labeled = sorted(((e, 'triplet') for e in energies_CIS_MO_triplet))
    energies_CIS_MO_labeled = sorted(energies_CIS_MO_singlet_labeled + energies_CIS_MO_triplet_labeled)
    print('CIS excitation energies (MO basis)')
    for i, (e, t) in enumerate(energies_CIS_MO_labeled, start=1):
        print(i, e, e * hartree_to_ev, t)

    # duplicate the triplet MO energies to match the SO results
    energies_CIS_MO_triplet_dup = []
    for e in energies_CIS_MO_triplet:
        for _ in range(3):
            energies_CIS_MO_triplet_dup.append(e)
    energies_CIS_MO = np.concatenate((energies_CIS_MO_singlet, energies_CIS_MO_triplet_dup), axis=0)
    idx_CIS_MO = energies_CIS_MO.argsort()
    energies_CIS_MO = energies_CIS_MO[idx_CIS_MO]

    assert energies_CIS_SO.shape == energies_CIS_MO.shape
    assert (energies_CIS_SO - energies_CIS_MO).all() == 0.0

    ## Time-Dependent Hartree-Fock (TDHF) / Random Phase Approximation (RPA)
    ## method 1
    A_SO = form_rpa_a_matrix_so(E_SO, TEI_SO, nsocc)
    B_SO = form_rpa_b_matrix_so(TEI_SO, nsocc)
    H_RPA_SO = np.bmat([[ A_SO,  B_SO],
                        [-B_SO, -A_SO]])
    assert matsym(A_SO) == 1
    assert matsym(B_SO) == 1
    assert matsym(H_RPA_SO) == 2
    energies_RPA_SO, eigvecs_RPA_SO = np.linalg.eig(H_RPA_SO)
    idx_RPA_SO = energies_RPA_SO.argsort()
    energies_RPA_SO = energies_RPA_SO[idx_RPA_SO].real
    eigvecs_RPA_SO = eigvecs_RPA_SO[idx_RPA_SO].real
    print('RPA excitation energies (SO basis), method 1')
    for i, e in enumerate(energies_RPA_SO, start=1):
        print(i, e, e * hartree_to_ev)

    ## method 2
    H_RPA_SO_reduced = np.dot(A_SO + B_SO, A_SO - B_SO)
    energies_RPA_SO_reduced, eigvecs_RPA_SO_reduced = np.linalg.eig(H_RPA_SO_reduced)
    idx_RPA_SO_reduced = energies_RPA_SO_reduced.argsort()
    energies_RPA_SO_reduced = np.sqrt(energies_RPA_SO_reduced[idx_RPA_SO_reduced].real)
    eigvecs_RPA_SO_reduced = eigvecs_RPA_SO_reduced[idx_RPA_SO_reduced].real
    print('RPA excitation energies (SO basis), method 2')
    for i, e in enumerate(energies_RPA_SO_reduced, start=1):
        print(i, e, e * hartree_to_ev)

    ## method 1, MO basis
    A_MO_singlet = form_rpa_a_matrix_mo_singlet(E_MO, TEI_MO, nocc)
    A_MO_triplet = form_rpa_a_matrix_mo_triplet(E_MO, TEI_MO, nocc)
    B_MO_singlet = form_rpa_b_matrix_mo_singlet(TEI_MO, nocc)
    B_MO_triplet = form_rpa_b_matrix_mo_triplet(TEI_MO, nocc)
    H_RPA_MO_singlet = np.bmat([[ A_MO_singlet,  B_MO_singlet],
                                [-B_MO_singlet, -A_MO_singlet]])
    H_RPA_MO_triplet = np.bmat([[ A_MO_triplet,  B_MO_triplet],
                                [-B_MO_triplet, -A_MO_triplet]])
    assert matsym(A_MO_singlet) == 1
    assert matsym(B_MO_singlet) == 1
    assert matsym(H_RPA_MO_singlet) == 2
    assert matsym(A_MO_triplet) == 1
    assert matsym(B_MO_triplet) == 1
    assert matsym(H_RPA_MO_triplet) == 2
    energies_RPA_MO_singlet, eigvecs_RPA_MO_singlet = np.linalg.eig(H_RPA_MO_singlet)
    energies_RPA_MO_triplet, eigvecs_RPA_MO_triplet = np.linalg.eig(H_RPA_MO_triplet)
    idx_RPA_MO_singlet = energies_RPA_MO_singlet.argsort()
    idx_RPA_MO_triplet = energies_RPA_MO_triplet.argsort()
    energies_RPA_MO_singlet = energies_RPA_MO_singlet[idx_RPA_MO_singlet].real
    energies_RPA_MO_triplet = energies_RPA_MO_triplet[idx_RPA_MO_triplet].real
    eigvecs_RPA_MO_singlet = eigvecs_RPA_MO_singlet[idx_RPA_MO_singlet].real
    eigvecs_RPA_MO_triplet = eigvecs_RPA_MO_triplet[idx_RPA_MO_triplet].real
    energies_RPA_MO_singlet_labeled = sorted(((e, 'singlet') for e in energies_RPA_MO_singlet))
    energies_RPA_MO_triplet_labeled = sorted(((e, 'triplet') for e in energies_RPA_MO_triplet))
    energies_RPA_MO_labeled = sorted(energies_RPA_MO_singlet_labeled + energies_RPA_MO_triplet_labeled)
    print('RPA excitation energies (MO basis), method 1')
    for i, (e, t) in enumerate(energies_RPA_MO_labeled, start=1):
        print(i, e, e * hartree_to_ev, t)

    # duplicate the triplet MO energies to match the SO results
    energies_RPA_MO_triplet_dup = []
    for e in energies_RPA_MO_triplet:
        for _ in range(3):
            energies_RPA_MO_triplet_dup.append(e)
    energies_RPA_MO = np.concatenate((energies_RPA_MO_singlet, energies_RPA_MO_triplet_dup), axis=0)
    idx_RPA_MO = energies_RPA_MO.argsort()
    energies_RPA_MO = energies_RPA_MO[idx_RPA_MO]

    assert energies_RPA_SO.shape == energies_RPA_MO.shape
    assert (energies_RPA_SO - energies_RPA_MO).all() == 0.0

    # superoverlap = np.bmat([[np.eye(nov), np.zeros(shape=(nov, nov))],
    #                         [np.zeros(shape=(nov, nov)), -np.eye(nov)]])
    # H_RPA_MO_singlet = np.bmat([[ A_MO_singlet,  B_MO_singlet],
    #                             [ B_MO_singlet,  A_MO_singlet]])
    # energies_RPA_MO_singlet, eigvecs_RPA_MO_singlet = np.linalg.eig(H_RPA_MO_singlet)
    # for i in range(len(energies_RPA_MO_singlet)):
    #     print(np.dot(eigvecs_RPA_MO_singlet[:, i].T, np.dot(superoverlap, eigvecs_RPA_MO_singlet[:, i])))
