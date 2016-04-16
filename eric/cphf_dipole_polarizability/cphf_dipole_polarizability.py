#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import sys

import numpy as np
import numpy.linalg as npl

from ..molecule import Molecule
from ..utils import print_mat
from ..utils import np_load


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--stub', default="h2o_sto3g")
    parser.add_argument('--nbasis', type=int, default=7)
    parser.add_argument('--nelec', type=int, default=10)
    parser.add_argument('--thresh-e', type=int, default=15)
    parser.add_argument('--thresh-d', type=int, default=10)
    parser.add_argument('--guess',
                        choices=('hcore', 'gwh'),
                        default='hcore',
                        help="""How should the guess for the initial MO
                        coefficients be obtained?""")

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


def population_analysis(mol, pop_mat, basis_function_indices):
    """Perform population analysis..."""

    charges = []

    for i in range(mol.size):

        # The basis function indices for each atom.
        bfi = basis_function_indices[i]

        # Take the trace of the "population" matrix block
        # corresponding to each individual atom.  Assuming that the
        # indices are in order, and the block is bounded by the first
        # and last elements of bfi. Is there a better way to do fancy
        # indexing here?
        tr = np.trace(pop_mat[bfi[0]:bfi[-1]+1,
                              bfi[0]:bfi[-1]+1])
        # Definition of the final charge.
        charge = mol.charges[i] - 2 * tr

        charges.append(charge)

    return np.asarray(charges)


def guess_gwh(mat_h, mat_s, cx=1.75):
    """From the core Hamiltonian and overlap matrices, form the matrix for
    the generalized Wolfsberg-Helmholz approximation (DOI:
    10.1063/1.1700580)

    The default value of 1.75 is from the Q-Chem 4.3 manual.
    """

    assert mat_h.shape == mat_s.shape
    nr, nc = mat_h.shape
    assert nr == nc
    mat_gwh = np.empty_like(mat_h)

    for mu in range(nr):
        for nu in range(nc):
            mat_gwh[mu, nu] = mat_s[mu, nu] * (mat_h[mu, mu] + mat_h[nu, nu])

    mat_gwh *= (cx / 2)

    return mat_gwh


def ao_mo_4index_smart_quarter(TEI_MO, TEI_AO, C):

    nbasis, norb = C.shape

    tmp1 = np.zeros(shape=(norb, nbasis, nbasis, nbasis))
    tmp2 = np.zeros(shape=(norb, norb, nbasis, nbasis))
    tmp3 = np.zeros(shape=(norb, norb, norb, nbasis))

    for p in range(norb):
        for mu in range(nbasis):
            tmp1[p, :, :, :] += C[mu, p] * TEI_AO[mu, :, :, :]
        for q in range(norb):
            for nu in range(nbasis):
                tmp2[p, q, :, :] += C[nu, q] * tmp1[p, nu, :, :]
            for r in range(norb):
                for lm in range(nbasis):
                    tmp3[p, q, r, :] += C[lm, r] * tmp2[p, q, lm, :]
                for s in range(norb):
                    for sg in range(nbasis):
                        TEI_MO[p, q, r, s] += C[sg, s] * tmp3[p, q, r, sg]

    return


def calc_mp2_energy(TEI_MO, E, nocc):

    E_MP2 = 0.0

    norb = len(E)

    for i in range(0, nocc):
        for j in range(0, nocc):
            for a in range(nocc, norb):
                for b in range(nocc, norb):
                    denominator = E[i] + E[j] - E[a] - E[b]
                    numerator = TEI_MO[i, a, j, b] * \
                                (2*TEI_MO[i, a, j, b] - TEI_MO[i, b, j, a])
                    E_MP2 += (numerator / denominator)

    return E_MP2


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


def form_hamiltonian_cis_mo(E_MO, TEI_MO, nocc):
    """Form the CIS Hamiltonian in the molecular orbital basis."""

    norb = E_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    H_CIS_MO = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    # <aj||ib> = <aj|ib> - <aj|bi> = (ai|jb) - (ab|ji)
                    # H_CIS_MO[ia, jb] = ((i == j) * E_MO[a + nocc, b + nocc]) - ((a == b) * E_MO[i, j]) + TEI_MO[a + nocc, i, j, b + nocc] - TEI_MO[a + nocc, b + nocc, j, i]
                    H_CIS_MO[ia, jb] = (i == j)*(a == b)*(E_MO[a + nocc, b + nocc] - E_MO[i, j]) + 4*TEI_MO[i, a + nocc, j, b + nocc] - TEI_MO[i, b + nocc, j, a + nocc] - TEI_MO[i, j, a + nocc, b + nocc]

    return H_CIS_MO


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


def repack_mat_to_vec_rows_cols(mat):
    nrows, ncols = mat.shape
    vec = np.empty((nrows*ncols))
    for i in range(nrows):
        for a in range(ncols):
            ia = i*ncols + a
            vec[ia] = mat[i, a]
    return vec


if __name__ == "__main__":

    args = getargs()
    stub = args.stub + "_"

    nelec = args.nelec
    nocc = nelec // 2
    norb = dim = nbasis = args.nbasis
    nvirt = norb - nocc
    nov = nocc * nvirt

    mol = Molecule(stub + "geom.dat")

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

    mat_h = mat_t + mat_v
    lam_s, l_s = npl.eigh(mat_s)
    lam_s = lam_s * np.eye(len(lam_s))
    lam_sqrt_inv = np.sqrt(npl.inv(lam_s))
    symm_orthog = np.dot(l_s, np.dot(lam_sqrt_inv, l_s.T))
    if args.guess == "hcore":
        f_prime = np.dot(symm_orthog.T, np.dot(mat_h, symm_orthog))
    elif args.guess == "gwh":
        mat_gwh = guess_gwh(mat_h, mat_s, cx=1.75)
        f_prime = np.dot(symm_orthog.T, np.dot(mat_gwh, symm_orthog))
    else:
        print("Invalid guess.", file=sys.stderr)
        sys.exit(1)
    eps, c_prime = npl.eigh(f_prime)
    eps = eps * np.eye(len(eps))
    c = np.dot(symm_orthog, c_prime)
    d = build_density(c, nocc)
    e_elec_new = calc_elec_energy(d, mat_h, mat_h)
    e_total = e_elec_new + e_nuc
    delta_e = e_total

    print(" Iter        E(elec)              E(tot)               Delta(E)             RMS(D)")
    print(" {:4d} {:20.12f} {:20.12f}".format(0, e_elec_new, e_total))
    t = " {:4d} {:20.12f} {:20.12f} {:20.12f} {:20.12f}".format

    f = np.empty(shape=(nbasis, nbasis))

    thresh_e = 10**(-args.thresh_e)
    thresh_d = 10**(-args.thresh_d)
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

    # At convergence, the Fock matrix should be diagonal in the MO
    # basis.
    f_mo = np.dot(c.T, np.dot(f, c))

    # Save things to disk for use in other routines.
    np.savez_compressed("H.npz", mat_h)
    np.savez_compressed("TEI_AO.npz", mat_eri)
    np.savez_compressed("C.npz", c)
    np.savez_compressed("F_MO.npz", f_mo)

    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    dipole_elec = 2 * np.array([np.sum(d * mat_dipole_x),
                                np.sum(d * mat_dipole_y),
                                np.sum(d * mat_dipole_z)])
    dipole_moment_elec = npl.norm(dipole_elec)

    dipole_nuc = mol.calc_dipole_nuc()
    dipole_moment_nuc = npl.norm(dipole_nuc)

    dipole_total = dipole_elec + dipole_nuc
    dipole_moment_total = npl.norm(dipole_total)

    print("Dipole components (electronic, a.u.):")
    print("X: {:20.12f}".format(dipole_elec[0]))
    print("Y: {:20.12f}".format(dipole_elec[1]))
    print("Z: {:20.12f}".format(dipole_elec[2]))

    print("Dipole components (nuclear, a.u.):")
    print("X: {:20.12f}".format(dipole_nuc[0]))
    print("Y: {:20.12f}".format(dipole_nuc[1]))
    print("Z: {:20.12f}".format(dipole_nuc[2]))

    print("Dipole components (total, a.u.):")
    print("X: {:20.12f}".format(dipole_total[0]))
    print("Y: {:20.12f}".format(dipole_total[1]))
    print("Z: {:20.12f}".format(dipole_total[2]))

    print("Dipole moment (a.u.):")
    print("electronic: {:20.12f}".format(dipole_moment_elec))
    print("nuclear   : {:20.12f}".format(dipole_moment_nuc))
    print("total     : {:20.12f}".format(dipole_moment_total))

    # This is cheating. How to determine this automatically without
    # any a priori knowledge of the basis set?
    basis_function_indices = [
        [0, 1, 2, 3, 4,],
        [5,],
        [6,],
    ]

    # Mulliken population analysis.

    mat_mulliken = np.dot(d, mat_s)
    charges_mulliken = population_analysis(mol, mat_mulliken, basis_function_indices)

    print("Population analysis (Mulliken):")
    print(" Charges:")
    for i in range(mol.size):
        print(" {:3d} {:3d} {:20.12f}".format(i + 1, mol.charges[i], charges_mulliken[i]))
    print(sum(charges_mulliken))
    print(" trace: {}".format(np.trace(mat_mulliken)))

    # Loewdin population analysis.

    mat_loewdin = np.dot(npl.inv(symm_orthog), np.dot(d, npl.inv(symm_orthog)))
    charges_loewdin = population_analysis(mol, mat_loewdin, basis_function_indices)

    print("Population analysis (Loewdin):")
    print(" Charges:")
    for i in range(mol.size):
        print(" {:3d} {:3d} {:20.12f}".format(i + 1, mol.charges[i], charges_loewdin[i]))
    print(sum(charges_loewdin))
    print(" trace: {}".format(np.trace(mat_loewdin)))

    # aliases
    E_MO = f_mo
    TEI_AO = mat_eri
    C = c
    H = mat_h

    TEI_MO = np.zeros(shape=TEI_AO.shape)
    ao_mo_4index_smart_quarter(TEI_MO, TEI_AO, C)
    np.savez_compressed('TEI_MO.npz', TEI_MO)
    E_MP2 = calc_mp2_energy(TEI_MO, np.diag(E_MO), nocc)
    print('E_MP2: {:20.12f}'.format(E_MP2))

    TEI_SO = np.zeros(shape=np.array(TEI_MO.shape) * 2)
    E_SO = make_spin_orbital_energies(E_MO)
    nsorb = TEI_SO.shape[0]
    nsocc = nocc * 2
    nsvir = nsorb - nsocc
    nsov = nsocc * nsvir
    print('nsorb: {} nsocc: {} nsvir: {} nsov: {}'.format(nsorb, nsocc, nsvir, nsov))
    # # Transform the two-electron integrals from the spatial MO-basis
    # # to the spin-orbital basis.
    # mo_so_4index(TEI_SO, TEI_MO)
    # # Build the Fock matrix in the SO basis.
    # F_SO = build_fock_spin_orbital(TEI_SO, H, nsocc)

    # H_CIS = np.zeros(shape=(nsov, nsov))
    # # Form the CIS Hamiltonian.
    # form_hamiltonian_cis(H_CIS, E_SO, TEI_SO, nsocc)
    # energies_CIS, eigvecs_CIS = np.linalg.eigh(H_CIS)
    # idx_CIS = energies_CIS.argsort()
    # energies_CIS = energies_CIS[idx_CIS]
    # eigvecs_CIS = eigvecs_CIS[idx_CIS]
    # print('CIS excitation energies (SO basis)')
    # hartree_to_ev = 27.211385
    # for i, e in enumerate(energies_CIS, start=1):
    #     print(i, e, e * hartree_to_ev)

    H_CIS_MO = form_hamiltonian_cis_mo(E_MO, TEI_MO, nocc)
    energies_CIS, eigvecs_CIS = np.linalg.eigh(H_CIS_MO)
    # print('CIS excitation energies (MO basis)')
    # for i, e in enumerate(energies_CIS, start=1):
    #     print(i, e, e * hartree_to_ev)

    # ## Spin-adapted CIS
    # H_CIS_singlet = np.zeros(shape=(nov, nov))
    # H_CIS_triplet = np.zeros(shape=(nov, nov))
    # form_hamiltonian_cis_singlet(H_CIS_singlet, E_MO, TEI_MO, nocc)
    # form_hamiltonian_cis_triplet(H_CIS_triplet, E_MO, TEI_MO, nocc)
    # energies_CIS_singlet, eigvecs_CIS_singlet = np.linalg.eigh(H_CIS_singlet)
    # energies_CIS_triplet, eigvecs_CIS_triplet = np.linalg.eigh(H_CIS_triplet)
    # _energies_CIS_singlet = sorted(((e, 'singlet') for e in energies_CIS_singlet))
    # _energies_CIS_triplet = sorted(((e, 'triplet') for e in energies_CIS_triplet))
    # energies_CIS = sorted(_energies_CIS_singlet + _energies_CIS_triplet)
    # print('CIS excitation energies (MO basis)')
    # for i, (e, t) in enumerate(energies_CIS, start=1):
    #     print(i, e, e * hartree_to_ev, t)

    # ## Time-Dependent Hartree-Fock (TDHF) / Random Phase Approximation (RPA)
    # A = form_rpa_a_matrix_so(E_SO, TEI_SO, nsocc)
    # B = form_rpa_b_matrix_so(TEI_SO, nsocc)
    # # Form the RPA supermatrix.
    # H_RPA = np.bmat([[ A,  B],
    #                  [-B, -A]])
    # eigvals_RPA, eigvecs_RPA = np.linalg.eig(H_RPA)
    # idx_RPA = eigvals_RPA.argsort()
    # eigvals_RPA = eigvals_RPA[idx_RPA].real
    # eigvecs_RPA = eigvecs_RPA[idx_RPA].real
    # print('RPA excitation energies (SO basis), method 1')
    # for i, e in enumerate(eigvals_RPA, start=1):
    #     print(i, e, e * hartree_to_ev)

    # H_RPA_reduced = np.dot(A + B, A - B)
    # eigvals_RPA_reduced, eigvecs_RPA_reduced = np.linalg.eig(H_RPA_reduced)
    # idx_RPA_reduced = eigvals_RPA_reduced.argsort()
    # eigvals_RPA_reduced = np.sqrt(eigvals_RPA_reduced[idx_RPA_reduced].real)
    # eigvecs_RPA_reduced = eigvecs_RPA_reduced[idx_RPA_reduced].real
    # print('RPA excitation energies (SO basis), method 2')
    # for i, e in enumerate(eigvals_RPA_reduced, start=1):
    #     print(i, e, e * hartree_to_ev)

    # Form the dipole integrals in the occ-virt MO basis.
    mat_dipole_x_MO_occ_virt = -2 * np.dot(C[:, :nocc].T, np.dot(mat_dipole_x, C[:, nocc:]))
    mat_dipole_y_MO_occ_virt = -2 * np.dot(C[:, :nocc].T, np.dot(mat_dipole_y, C[:, nocc:]))
    mat_dipole_z_MO_occ_virt = -2 * np.dot(C[:, :nocc].T, np.dot(mat_dipole_z, C[:, nocc:]))

    # Re-pack the integrals as vectors.
    gp_dipole_x = repack_mat_to_vec_rows_cols(mat_dipole_x_MO_occ_virt)
    gp_dipole_y = repack_mat_to_vec_rows_cols(mat_dipole_y_MO_occ_virt)
    gp_dipole_z = repack_mat_to_vec_rows_cols(mat_dipole_z_MO_occ_virt)

    # Explicitly invert the orbital Hessian.
    H_CIS_MO_inv = np.linalg.inv(H_CIS_MO)

    # Form the dipole response vectors.
    rspvec_dipole_x = np.dot(H_CIS_MO_inv, gp_dipole_x)
    rspvec_dipole_y = np.dot(H_CIS_MO_inv, gp_dipole_y)
    rspvec_dipole_z = np.dot(H_CIS_MO_inv, gp_dipole_z)

    polarizability = np.empty((3, 3))
    polarizability[0, 0] = np.dot(rspvec_dipole_x, gp_dipole_x)
    polarizability[0, 1] = np.dot(rspvec_dipole_x, gp_dipole_y)
    polarizability[0, 2] = np.dot(rspvec_dipole_x, gp_dipole_z)
    polarizability[1, 0] = np.dot(rspvec_dipole_y, gp_dipole_x)
    polarizability[1, 1] = np.dot(rspvec_dipole_y, gp_dipole_y)
    polarizability[1, 2] = np.dot(rspvec_dipole_y, gp_dipole_z)
    polarizability[2, 0] = np.dot(rspvec_dipole_z, gp_dipole_x)
    polarizability[2, 1] = np.dot(rspvec_dipole_z, gp_dipole_y)
    polarizability[2, 2] = np.dot(rspvec_dipole_z, gp_dipole_z)
    polarizability *= -1.0
    print('Dipole polarizability (a.u., explicit)')
    print(polarizability)
