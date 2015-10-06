#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.linalg as npl

from ..utils import print_mat


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--stub', default="h2o_sto3g")
    parser.add_argument('--nelec', type=int, default=10)

    args = parser.parse_args()

    return args


def parse_file_1(filename):
    with open(filename) as fh:
        val = float(fh.readline())
    return val


def parse_int_file_2(filename, dim):
    mat = np.zeros(shape=(dim, dim))
    with open(filename) as fh:
        contents = fh.readlines()
    for line in contents:
        mu, nu, intval = map(float, line.split())
        mat[mu-1, nu-1] = mat[nu-1, mu-1] = intval
    return mat


def parse_int_file_4(filename, dim):
    # be very inefficient with how we store these for now -- use all 4
    # indices
    mat = np.zeros(shape=(dim, dim, dim, dim))
    with open(filename) as fh:
        contents = fh.readlines()
    for line in contents:
        mu, nu, lm, sg, intval = map(float, line.split())
        mu, nu, lm, sg = mu - 1, nu - 1, lm - 1, sg - 1
        mat[mu, nu, lm, sg] = \
            mat[mu, nu, sg, lm] = \
            mat[nu, mu, lm, sg] = \
            mat[nu, mu, sg, lm] = \
            mat[lm, sg, mu, nu] = \
            mat[lm, sg, nu, mu] = \
            mat[sg, lm, mu, nu] = \
            mat[sg, lm, nu, mu] = intval
    return mat

## This doesn't work like it does for the Fock build.

# def build_density(P, C, nbasis, nocc):

#     for mu in range(nbasis):
#         for nu in range(nbasis):
#             for m in range(nocc):
#                 P[mu, nu] += C[mu, m] * C[nu, m]

#     return

def build_density(C, nocc):
    """Form the density matrix from contraction over the occupied columns
    of the MO coefficient matrix.
    """

    return np.dot(C[:, :nocc], C[:, :nocc].T)


def calc_elec_energy(P, H, F):
    """Calculate the electronic energy from contracting the density matrix
    with the one- (core Hamiltonian) and two- (Fock matrix) electron
    components of the Hamiltonian.
    """

    return np.sum(P * (H + F))


def build_fock(F, P, H, ERI, nbasis):
    """Build the Fock matrix in-place."""

    for mu in range(H.shape[0]):
        for nu in range(H.shape[1]):
            F[mu, nu] = H[mu, nu]
            for lm in range(P.shape[0]):
                for sg in range(P.shape[1]):
                    F[mu, nu] += (P[lm, sg] * (2*ERI[mu, nu, lm, sg] -
                                               ERI[mu, lm, nu, sg]))

    return


def rmsd_density(D_new, D_old):
    """Calculate the root mean square deviation between two density
    matrices.
    """

    return np.sqrt(np.sum((D_new - D_old)**2))


if __name__ == "__main__":

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    dim = 7
    nbasis = dim

    stub = args.stub + "_"

    filename_enuc = stub + "enuc.dat"
    filename_s = stub + "s.dat"
    filename_t = stub + "t.dat"
    filename_v = stub + "v.dat"
    filename_eri = stub + "eri.dat"

    e_nuc = parse_file_1(filename_enuc)
    mat_s = parse_int_file_2(filename_s, dim)
    mat_t = parse_int_file_2(filename_t, dim)
    mat_v = parse_int_file_2(filename_v, dim)
    mat_eri = parse_int_file_4(filename_eri, dim)

    print("Nuclear repulsion energy = {}\n".format(e_nuc))
    print("Overlap Integrals:")
    print_mat(mat_s)
    print("Kinetic-Energy Integrals:")
    print_mat(mat_t)
    print("Nuclear Attraction Integrals:")
    print_mat(mat_v)

    mat_h = mat_t + mat_v
    print("Core Hamiltonian:")
    print_mat(mat_h)
    lam_s, l_s = npl.eigh(mat_s)
    lam_s = lam_s * np.eye(len(lam_s))
    lam_sqrt_inv = np.sqrt(npl.inv(lam_s))
    symm_orthog = np.dot(l_s, np.dot(lam_sqrt_inv, l_s.T))
    print("S^-1/2 Matrix:")
    print_mat(symm_orthog)
    f_prime = np.dot(symm_orthog.T, np.dot(mat_h, symm_orthog))
    print("Initial F' Matrix:")
    print_mat(f_prime)
    eps, c_prime = npl.eigh(f_prime)
    eps = eps * np.eye(len(eps))
    c = np.dot(symm_orthog, c_prime)
    print("Initial C Matrix:")
    print_mat(c)
    d = build_density(c, nocc)
    print("Initial Density Matrix:")
    print_mat(d)
    e_elec_new = calc_elec_energy(d, mat_h, mat_h)
    e_total = e_elec_new + e_nuc
    delta_e = e_total

    print(" Iter        E(elec)              E(tot)               Delta(E)             RMS(D)")
    print(" {:4d} {:20.12f} {:20.12f}".format(0, e_elec_new, e_total))
    t = " {:4d} {:20.12f} {:20.12f} {:20.12f} {:20.12f}".format

    f = np.empty(shape=(nbasis, nbasis))

    thresh_e = 1.0e-15
    thresh_d = 1.0e-7
    iteration = 1
    max_iterations = 1024
    rmsd_d = 99999.9

    while iteration < max_iterations:

        build_fock(f, d, mat_h, mat_eri, nbasis)
        f_prime = np.dot(symm_orthog.T, np.dot(f, symm_orthog))
        eps, c_prime = npl.eigh(f_prime)
        eps = eps * np.eye(len(eps))
        c = np.dot(symm_orthog, c_prime)
        d_old = d
        d = build_density(c, nocc)
        e_elec_old = e_elec_new
        e_elec_new = calc_elec_energy(d, mat_h, f)
        e_tot = e_elec_new + e_nuc
        if iteration == 1:
            print("Fock Matrix:")
            print_mat(f)
            print(t(iteration, e_elec_new, e_tot, 0.0, 0.0))
        else:
            print(t(iteration, e_elec_new, e_tot, delta_e, rmsd_d))
        delta_e = e_elec_new - e_elec_old
        rmsd_d = rmsd_density(d, d_old)
        if (delta_e < thresh_e) and (rmsd_d < thresh_d):
            print("Convergence achieved.")
            break
        f = f_prime

        iteration += 1
