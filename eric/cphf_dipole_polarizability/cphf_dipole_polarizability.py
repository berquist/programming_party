import sys

import numpy as np
import numpy.linalg as npl

from ..molecule import Molecule
from ..utils import print_mat
from ..utils import np_load

from ..project3.project3 import \
    (parse_file_1, parse_int_file_2, parse_int_file_4, build_density,
     calc_elec_energy, build_fock, rmsd_density)
from ..project4.project4 import \
    (ao_mo_4index_smart_quarter, calc_mp2_energy)
from ..project12.project12 import \
    (make_spin_orbital_energies,
     form_hamiltonian_cis_mo_singlet, form_hamiltonian_cis_mo_triplet,
     form_rpa_a_matrix_mo_singlet, form_rpa_a_matrix_mo_triplet,
     form_rpa_b_matrix_mo_singlet, form_rpa_b_matrix_mo_triplet)


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

    # Form the dipole integrals in the occ-virt MO basis.
    mat_dipole_x_MO_occ_virt = np.dot(C[:, :nocc].T, np.dot(mat_dipole_x, C[:, nocc:]))
    mat_dipole_y_MO_occ_virt = np.dot(C[:, :nocc].T, np.dot(mat_dipole_y, C[:, nocc:]))
    mat_dipole_z_MO_occ_virt = np.dot(C[:, :nocc].T, np.dot(mat_dipole_z, C[:, nocc:]))

    # Re-pack the integrals as vectors.
    gp_dipole_x = repack_mat_to_vec_rows_cols(mat_dipole_x_MO_occ_virt)
    gp_dipole_y = repack_mat_to_vec_rows_cols(mat_dipole_y_MO_occ_virt)
    gp_dipole_z = repack_mat_to_vec_rows_cols(mat_dipole_z_MO_occ_virt)
    gp_dipole_x_super = np.concatenate((gp_dipole_x, gp_dipole_x), axis=0)
    gp_dipole_y_super = np.concatenate((gp_dipole_y, gp_dipole_y), axis=0)
    gp_dipole_z_super = np.concatenate((gp_dipole_z, gp_dipole_z), axis=0)

    # Form the orbital Hessian for each Hamiltonian and spin case.
    H_CIS_MO_singlet = np.zeros(shape=(nov, nov))
    H_CIS_MO_triplet = np.zeros(shape=(nov, nov))
    form_hamiltonian_cis_mo_singlet(H_CIS_MO_singlet, E_MO, TEI_MO, nocc)
    form_hamiltonian_cis_mo_triplet(H_CIS_MO_triplet, E_MO, TEI_MO, nocc)
    A_MO_singlet = form_rpa_a_matrix_mo_singlet(E_MO, TEI_MO, nocc)
    A_MO_triplet = form_rpa_a_matrix_mo_triplet(E_MO, TEI_MO, nocc)
    B_MO_singlet = form_rpa_b_matrix_mo_singlet(TEI_MO, nocc)
    B_MO_triplet = form_rpa_b_matrix_mo_triplet(TEI_MO, nocc)
    H_RPA_MO_singlet = np.bmat([[ A_MO_singlet,  B_MO_singlet],
                                [ B_MO_singlet,  A_MO_singlet]])
    H_RPA_MO_triplet = np.bmat([[ A_MO_triplet,  B_MO_triplet],
                                [ B_MO_triplet,  A_MO_triplet]])
    ApB_MO_singlet = A_MO_singlet + B_MO_singlet
    AmB_MO_singlet = A_MO_singlet - B_MO_singlet
    ApB_MO_triplet = A_MO_triplet + B_MO_triplet
    AmB_MO_triplet = A_MO_triplet - B_MO_triplet

    # np.testing.assert_allclose(H_CIS_MO_singlet, A_MO_singlet)
    # np.testing.assert_allclose(H_CIS_MO_triplet, A_MO_triplet)

    # Explicitly invert the orbital Hessian for each Hamiltonian and
    # spin case.
    H_CIS_MO_singlet_inv = np.linalg.inv(H_CIS_MO_singlet)
    H_CIS_MO_triplet_inv = np.linalg.inv(H_CIS_MO_triplet)
    H_RPA_MO_singlet_inv = np.linalg.inv(H_RPA_MO_singlet)
    H_RPA_MO_triplet_inv = np.linalg.inv(H_RPA_MO_triplet)
    ApB_MO_singlet_inv = np.linalg.inv(ApB_MO_singlet)
    AmB_MO_singlet_inv = np.linalg.inv(AmB_MO_singlet)
    ApB_MO_triplet_inv = np.linalg.inv(ApB_MO_triplet)
    AmB_MO_triplet_inv = np.linalg.inv(AmB_MO_triplet)

    # Form the dipole response vectors.
    rspvec_A_singlet_dipole_x = np.dot(H_CIS_MO_singlet_inv, gp_dipole_x)
    rspvec_A_singlet_dipole_y = np.dot(H_CIS_MO_singlet_inv, gp_dipole_y)
    rspvec_A_singlet_dipole_z = np.dot(H_CIS_MO_singlet_inv, gp_dipole_z)
    rspvec_A_triplet_dipole_x = np.dot(H_CIS_MO_triplet_inv, gp_dipole_x)
    rspvec_A_triplet_dipole_y = np.dot(H_CIS_MO_triplet_inv, gp_dipole_y)
    rspvec_A_triplet_dipole_z = np.dot(H_CIS_MO_triplet_inv, gp_dipole_z)
    rspvec_ApB_singlet_dipole_x = np.dot(ApB_MO_singlet_inv, gp_dipole_x)
    rspvec_ApB_singlet_dipole_y = np.dot(ApB_MO_singlet_inv, gp_dipole_y)
    rspvec_ApB_singlet_dipole_z = np.dot(ApB_MO_singlet_inv, gp_dipole_z)
    rspvec_AmB_singlet_dipole_x = np.dot(AmB_MO_singlet_inv, gp_dipole_x)
    rspvec_AmB_singlet_dipole_y = np.dot(AmB_MO_singlet_inv, gp_dipole_y)
    rspvec_AmB_singlet_dipole_z = np.dot(AmB_MO_singlet_inv, gp_dipole_z)
    rspvec_ApB_triplet_dipole_x = np.dot(ApB_MO_triplet_inv, gp_dipole_x)
    rspvec_ApB_triplet_dipole_y = np.dot(ApB_MO_triplet_inv, gp_dipole_y)
    rspvec_ApB_triplet_dipole_z = np.dot(ApB_MO_triplet_inv, gp_dipole_z)
    rspvec_AmB_triplet_dipole_x = np.dot(AmB_MO_triplet_inv, gp_dipole_x)
    rspvec_AmB_triplet_dipole_y = np.dot(AmB_MO_triplet_inv, gp_dipole_y)
    rspvec_AmB_triplet_dipole_z = np.dot(AmB_MO_triplet_inv, gp_dipole_z)
    rspvec_RPA_singlet_dipole_x = np.dot(H_RPA_MO_singlet_inv, gp_dipole_x_super)
    rspvec_RPA_singlet_dipole_y = np.dot(H_RPA_MO_singlet_inv, gp_dipole_y_super)
    rspvec_RPA_singlet_dipole_z = np.dot(H_RPA_MO_singlet_inv, gp_dipole_z_super)
    rspvec_RPA_triplet_dipole_x = np.dot(H_RPA_MO_triplet_inv, gp_dipole_x_super)
    rspvec_RPA_triplet_dipole_y = np.dot(H_RPA_MO_triplet_inv, gp_dipole_y_super)
    rspvec_RPA_triplet_dipole_z = np.dot(H_RPA_MO_triplet_inv, gp_dipole_z_super)

    # TDA singlet
    polarizability_A_singlet = np.empty((3, 3))
    polarizability_A_singlet[0, 0] = np.dot(rspvec_A_singlet_dipole_x, gp_dipole_x)
    polarizability_A_singlet[0, 1] = np.dot(rspvec_A_singlet_dipole_x, gp_dipole_y)
    polarizability_A_singlet[0, 2] = np.dot(rspvec_A_singlet_dipole_x, gp_dipole_z)
    polarizability_A_singlet[1, 0] = np.dot(rspvec_A_singlet_dipole_y, gp_dipole_x)
    polarizability_A_singlet[1, 1] = np.dot(rspvec_A_singlet_dipole_y, gp_dipole_y)
    polarizability_A_singlet[1, 2] = np.dot(rspvec_A_singlet_dipole_y, gp_dipole_z)
    polarizability_A_singlet[2, 0] = np.dot(rspvec_A_singlet_dipole_z, gp_dipole_x)
    polarizability_A_singlet[2, 1] = np.dot(rspvec_A_singlet_dipole_z, gp_dipole_y)
    polarizability_A_singlet[2, 2] = np.dot(rspvec_A_singlet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, A singlet [a.u] -> TDA/CIS')
    print(polarizability_A_singlet * 4)

    # TDA triplet
    polarizability_A_triplet = np.empty((3, 3))
    polarizability_A_triplet[0, 0] = np.dot(rspvec_A_triplet_dipole_x, gp_dipole_x)
    polarizability_A_triplet[0, 1] = np.dot(rspvec_A_triplet_dipole_x, gp_dipole_y)
    polarizability_A_triplet[0, 2] = np.dot(rspvec_A_triplet_dipole_x, gp_dipole_z)
    polarizability_A_triplet[1, 0] = np.dot(rspvec_A_triplet_dipole_y, gp_dipole_x)
    polarizability_A_triplet[1, 1] = np.dot(rspvec_A_triplet_dipole_y, gp_dipole_y)
    polarizability_A_triplet[1, 2] = np.dot(rspvec_A_triplet_dipole_y, gp_dipole_z)
    polarizability_A_triplet[2, 0] = np.dot(rspvec_A_triplet_dipole_z, gp_dipole_x)
    polarizability_A_triplet[2, 1] = np.dot(rspvec_A_triplet_dipole_z, gp_dipole_y)
    polarizability_A_triplet[2, 2] = np.dot(rspvec_A_triplet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, A triplet [a.u] -> TDA/CIS')
    print(polarizability_A_triplet * 4)

    # RPA singlet
    polarizability_ApB_singlet = np.empty((3, 3))
    polarizability_ApB_singlet[0, 0] = np.dot(rspvec_ApB_singlet_dipole_x, gp_dipole_x)
    polarizability_ApB_singlet[0, 1] = np.dot(rspvec_ApB_singlet_dipole_x, gp_dipole_y)
    polarizability_ApB_singlet[0, 2] = np.dot(rspvec_ApB_singlet_dipole_x, gp_dipole_z)
    polarizability_ApB_singlet[1, 0] = np.dot(rspvec_ApB_singlet_dipole_y, gp_dipole_x)
    polarizability_ApB_singlet[1, 1] = np.dot(rspvec_ApB_singlet_dipole_y, gp_dipole_y)
    polarizability_ApB_singlet[1, 2] = np.dot(rspvec_ApB_singlet_dipole_y, gp_dipole_z)
    polarizability_ApB_singlet[2, 0] = np.dot(rspvec_ApB_singlet_dipole_z, gp_dipole_x)
    polarizability_ApB_singlet[2, 1] = np.dot(rspvec_ApB_singlet_dipole_z, gp_dipole_y)
    polarizability_ApB_singlet[2, 2] = np.dot(rspvec_ApB_singlet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, (A + B) singlet [a.u] -> RPA (real operators)')
    print(polarizability_ApB_singlet * 4)

    # RPA triplet
    polarizability_ApB_triplet = np.empty((3, 3))
    polarizability_ApB_triplet[0, 0] = np.dot(rspvec_ApB_triplet_dipole_x, gp_dipole_x)
    polarizability_ApB_triplet[0, 1] = np.dot(rspvec_ApB_triplet_dipole_x, gp_dipole_y)
    polarizability_ApB_triplet[0, 2] = np.dot(rspvec_ApB_triplet_dipole_x, gp_dipole_z)
    polarizability_ApB_triplet[1, 0] = np.dot(rspvec_ApB_triplet_dipole_y, gp_dipole_x)
    polarizability_ApB_triplet[1, 1] = np.dot(rspvec_ApB_triplet_dipole_y, gp_dipole_y)
    polarizability_ApB_triplet[1, 2] = np.dot(rspvec_ApB_triplet_dipole_y, gp_dipole_z)
    polarizability_ApB_triplet[2, 0] = np.dot(rspvec_ApB_triplet_dipole_z, gp_dipole_x)
    polarizability_ApB_triplet[2, 1] = np.dot(rspvec_ApB_triplet_dipole_z, gp_dipole_y)
    polarizability_ApB_triplet[2, 2] = np.dot(rspvec_ApB_triplet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, (A + B) triplet [a.u] -> RPA (real operators)')
    print(polarizability_ApB_triplet * 4)

    polarizability_AmB_singlet = np.empty((3, 3))
    polarizability_AmB_singlet[0, 0] = np.dot(rspvec_AmB_singlet_dipole_x, gp_dipole_x)
    polarizability_AmB_singlet[0, 1] = np.dot(rspvec_AmB_singlet_dipole_x, gp_dipole_y)
    polarizability_AmB_singlet[0, 2] = np.dot(rspvec_AmB_singlet_dipole_x, gp_dipole_z)
    polarizability_AmB_singlet[1, 0] = np.dot(rspvec_AmB_singlet_dipole_y, gp_dipole_x)
    polarizability_AmB_singlet[1, 1] = np.dot(rspvec_AmB_singlet_dipole_y, gp_dipole_y)
    polarizability_AmB_singlet[1, 2] = np.dot(rspvec_AmB_singlet_dipole_y, gp_dipole_z)
    polarizability_AmB_singlet[2, 0] = np.dot(rspvec_AmB_singlet_dipole_z, gp_dipole_x)
    polarizability_AmB_singlet[2, 1] = np.dot(rspvec_AmB_singlet_dipole_z, gp_dipole_y)
    polarizability_AmB_singlet[2, 2] = np.dot(rspvec_AmB_singlet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, (A - B) singlet [a.u] -> RPA (imaginary operators)')
    print(polarizability_AmB_singlet * 4)

    polarizability_AmB_triplet = np.empty((3, 3))
    polarizability_AmB_triplet[0, 0] = np.dot(rspvec_AmB_triplet_dipole_x, gp_dipole_x)
    polarizability_AmB_triplet[0, 1] = np.dot(rspvec_AmB_triplet_dipole_x, gp_dipole_y)
    polarizability_AmB_triplet[0, 2] = np.dot(rspvec_AmB_triplet_dipole_x, gp_dipole_z)
    polarizability_AmB_triplet[1, 0] = np.dot(rspvec_AmB_triplet_dipole_y, gp_dipole_x)
    polarizability_AmB_triplet[1, 1] = np.dot(rspvec_AmB_triplet_dipole_y, gp_dipole_y)
    polarizability_AmB_triplet[1, 2] = np.dot(rspvec_AmB_triplet_dipole_y, gp_dipole_z)
    polarizability_AmB_triplet[2, 0] = np.dot(rspvec_AmB_triplet_dipole_z, gp_dipole_x)
    polarizability_AmB_triplet[2, 1] = np.dot(rspvec_AmB_triplet_dipole_z, gp_dipole_y)
    polarizability_AmB_triplet[2, 2] = np.dot(rspvec_AmB_triplet_dipole_z, gp_dipole_z)
    print('Static dipole polarizability, (A - B) triplet [a.u] -> RPA (imaginary operators)')
    print(polarizability_AmB_triplet * 4)

    polarizability_RPA_singlet = np.empty((3, 3))
    polarizability_RPA_singlet[0, 0] = np.dot(rspvec_RPA_singlet_dipole_x, gp_dipole_x_super)
    polarizability_RPA_singlet[0, 1] = np.dot(rspvec_RPA_singlet_dipole_x, gp_dipole_y_super)
    polarizability_RPA_singlet[0, 2] = np.dot(rspvec_RPA_singlet_dipole_x, gp_dipole_z_super)
    polarizability_RPA_singlet[1, 0] = np.dot(rspvec_RPA_singlet_dipole_y, gp_dipole_x_super)
    polarizability_RPA_singlet[1, 1] = np.dot(rspvec_RPA_singlet_dipole_y, gp_dipole_y_super)
    polarizability_RPA_singlet[1, 2] = np.dot(rspvec_RPA_singlet_dipole_y, gp_dipole_z_super)
    polarizability_RPA_singlet[2, 0] = np.dot(rspvec_RPA_singlet_dipole_z, gp_dipole_x_super)
    polarizability_RPA_singlet[2, 1] = np.dot(rspvec_RPA_singlet_dipole_z, gp_dipole_y_super)
    polarizability_RPA_singlet[2, 2] = np.dot(rspvec_RPA_singlet_dipole_z, gp_dipole_z_super)
    print('Static dipole polarizability, (A + B) singlet [a.u] -> RPA (supervector)')
    # TODO why divide by 2 again?
    print(polarizability_RPA_singlet * 4 / 2.0)
    # print(np.sqrt(polarizability_RPA_singlet))

    polarizability_RPA_triplet = np.empty((3, 3))
    polarizability_RPA_triplet[0, 0] = np.dot(rspvec_RPA_triplet_dipole_x, gp_dipole_x_super)
    polarizability_RPA_triplet[0, 1] = np.dot(rspvec_RPA_triplet_dipole_x, gp_dipole_y_super)
    polarizability_RPA_triplet[0, 2] = np.dot(rspvec_RPA_triplet_dipole_x, gp_dipole_z_super)
    polarizability_RPA_triplet[1, 0] = np.dot(rspvec_RPA_triplet_dipole_y, gp_dipole_x_super)
    polarizability_RPA_triplet[1, 1] = np.dot(rspvec_RPA_triplet_dipole_y, gp_dipole_y_super)
    polarizability_RPA_triplet[1, 2] = np.dot(rspvec_RPA_triplet_dipole_y, gp_dipole_z_super)
    polarizability_RPA_triplet[2, 0] = np.dot(rspvec_RPA_triplet_dipole_z, gp_dipole_x_super)
    polarizability_RPA_triplet[2, 1] = np.dot(rspvec_RPA_triplet_dipole_z, gp_dipole_y_super)
    polarizability_RPA_triplet[2, 2] = np.dot(rspvec_RPA_triplet_dipole_z, gp_dipole_z_super)
    print('Static dipole polarizability, (A + B) triplet [a.u] -> RPA (supervector)')
    print(polarizability_RPA_triplet * 4 / 2.0)
    # print(np.sqrt(polarizability_RPA_triplet))

    # This doesn't work when not doing the supervector formalism!
    # For the dynamic polarizability, the frequency (in a.u.) gets
    # subtracted from the orbital Hessian diagonal.
    # H_CIS_MO_singlet_omega = np.zeros(shape=(nov, nov))
    # form_hamiltonian_cis_mo_singlet(H_CIS_MO_singlet_omega, E_MO, TEI_MO, nocc)
    # H_CIS_MO_singlet_omega -= (omega * np.eye(nov))
    # H_CIS_MO_singlet_omega_inv = np.linalg.inv(H_CIS_MO_singlet_omega)
    # rspvec_A_singlet_omega_dipole_x = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_x)
    # rspvec_A_singlet_omega_dipole_y = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_y)
    # rspvec_A_singlet_omega_dipole_z = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_z)
    # polarizability_A_singlet_omega = np.empty((3, 3))
    # polarizability_A_singlet_omega[0, 0] = np.dot(rspvec_A_singlet_omega_dipole_x, gp_dipole_x)
    # polarizability_A_singlet_omega[0, 1] = np.dot(rspvec_A_singlet_omega_dipole_x, gp_dipole_y)
    # polarizability_A_singlet_omega[0, 2] = np.dot(rspvec_A_singlet_omega_dipole_x, gp_dipole_z)
    # polarizability_A_singlet_omega[1, 0] = np.dot(rspvec_A_singlet_omega_dipole_y, gp_dipole_x)
    # polarizability_A_singlet_omega[1, 1] = np.dot(rspvec_A_singlet_omega_dipole_y, gp_dipole_y)
    # polarizability_A_singlet_omega[1, 2] = np.dot(rspvec_A_singlet_omega_dipole_y, gp_dipole_z)
    # polarizability_A_singlet_omega[2, 0] = np.dot(rspvec_A_singlet_omega_dipole_z, gp_dipole_x)
    # polarizability_A_singlet_omega[2, 1] = np.dot(rspvec_A_singlet_omega_dipole_z, gp_dipole_y)
    # polarizability_A_singlet_omega[2, 2] = np.dot(rspvec_A_singlet_omega_dipole_z, gp_dipole_z)
    # print(f'Dynamic dipole polarizability @ {omega:f} a.u. -> TDA singlet')
    # print(polarizability_A_singlet_omega * 4)

    superoverlap = np.bmat([[np.eye(nov), np.zeros(shape=(nov, nov))],
                            [np.zeros(shape=(nov, nov)), -np.eye(nov)]])
    # frequencies of incident radiation in a.u.
    omegas = (0.0, 0.02, 0.06, 0.1)

    for omega in omegas:
        fd_superoverlap = omega * superoverlap
        H_RPA_MO_singlet_omega = H_RPA_MO_singlet - fd_superoverlap
        H_RPA_MO_singlet_omega_inv = np.linalg.inv(H_RPA_MO_singlet_omega)
        rspvec_RPA_singlet_omega_dipole_x = np.dot(H_RPA_MO_singlet_omega_inv, gp_dipole_x_super)
        rspvec_RPA_singlet_omega_dipole_y = np.dot(H_RPA_MO_singlet_omega_inv, gp_dipole_y_super)
        rspvec_RPA_singlet_omega_dipole_z = np.dot(H_RPA_MO_singlet_omega_inv, gp_dipole_z_super)
        polarizability_RPA_singlet_omega = np.empty((3, 3))
        polarizability_RPA_singlet_omega[0, 0] = np.dot(rspvec_RPA_singlet_omega_dipole_x, gp_dipole_x_super)
        polarizability_RPA_singlet_omega[0, 1] = np.dot(rspvec_RPA_singlet_omega_dipole_x, gp_dipole_y_super)
        polarizability_RPA_singlet_omega[0, 2] = np.dot(rspvec_RPA_singlet_omega_dipole_x, gp_dipole_z_super)
        polarizability_RPA_singlet_omega[1, 0] = np.dot(rspvec_RPA_singlet_omega_dipole_y, gp_dipole_x_super)
        polarizability_RPA_singlet_omega[1, 1] = np.dot(rspvec_RPA_singlet_omega_dipole_y, gp_dipole_y_super)
        polarizability_RPA_singlet_omega[1, 2] = np.dot(rspvec_RPA_singlet_omega_dipole_y, gp_dipole_z_super)
        polarizability_RPA_singlet_omega[2, 0] = np.dot(rspvec_RPA_singlet_omega_dipole_z, gp_dipole_x_super)
        polarizability_RPA_singlet_omega[2, 1] = np.dot(rspvec_RPA_singlet_omega_dipole_z, gp_dipole_y_super)
        polarizability_RPA_singlet_omega[2, 2] = np.dot(rspvec_RPA_singlet_omega_dipole_z, gp_dipole_z_super)
        print(f'Dynamic dipole polarizability @ {omega:f} a.u. -> RPA singlet supervector')
        print(polarizability_RPA_singlet_omega * 4 / 2)

        # X_dipole_x = rspvec_RPA_singlet_omega_dipole_x.T[:nov, :]
        # Y_dipole_x = rspvec_RPA_singlet_omega_dipole_x.T[nov:, :]
        # print(X_dipole_x.T)
        # print(Y_dipole_x.T)

    for omega in omegas:
        fd_superoverlap = omega * superoverlap
        H_CIS_MO_singlet_super = np.bmat([[ A_MO_singlet,  np.zeros(shape=(nov, nov))],
                                          [ np.zeros(shape=(nov, nov)),  A_MO_singlet]])
        H_CIS_MO_singlet_omega = H_CIS_MO_singlet_super - fd_superoverlap
        H_CIS_MO_singlet_omega_inv = np.linalg.inv(H_CIS_MO_singlet_omega)
        rspvec_CIS_singlet_omega_dipole_x = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_x_super)
        rspvec_CIS_singlet_omega_dipole_y = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_y_super)
        rspvec_CIS_singlet_omega_dipole_z = np.dot(H_CIS_MO_singlet_omega_inv, gp_dipole_z_super)
        polarizability_CIS_singlet_omega = np.empty((3, 3))
        polarizability_CIS_singlet_omega[0, 0] = np.dot(rspvec_CIS_singlet_omega_dipole_x, gp_dipole_x_super)
        polarizability_CIS_singlet_omega[0, 1] = np.dot(rspvec_CIS_singlet_omega_dipole_x, gp_dipole_y_super)
        polarizability_CIS_singlet_omega[0, 2] = np.dot(rspvec_CIS_singlet_omega_dipole_x, gp_dipole_z_super)
        polarizability_CIS_singlet_omega[1, 0] = np.dot(rspvec_CIS_singlet_omega_dipole_y, gp_dipole_x_super)
        polarizability_CIS_singlet_omega[1, 1] = np.dot(rspvec_CIS_singlet_omega_dipole_y, gp_dipole_y_super)
        polarizability_CIS_singlet_omega[1, 2] = np.dot(rspvec_CIS_singlet_omega_dipole_y, gp_dipole_z_super)
        polarizability_CIS_singlet_omega[2, 0] = np.dot(rspvec_CIS_singlet_omega_dipole_z, gp_dipole_x_super)
        polarizability_CIS_singlet_omega[2, 1] = np.dot(rspvec_CIS_singlet_omega_dipole_z, gp_dipole_y_super)
        polarizability_CIS_singlet_omega[2, 2] = np.dot(rspvec_CIS_singlet_omega_dipole_z, gp_dipole_z_super)
        print(f'Dynamic dipole polarizability @ {omega:f} a.u. -> CIS singlet supervector')
        print(polarizability_CIS_singlet_omega * 4 / 2)

        # X_dipole_x = rspvec_CIS_singlet_omega_dipole_x.T[:nov, :]
        # Y_dipole_x = rspvec_CIS_singlet_omega_dipole_x.T[nov:, :]
        # print(X_dipole_x.T)
        # print(Y_dipole_x.T)

    H_RPA_MO_singlet_flip = np.bmat([[  A_MO_singlet,   B_MO_singlet],
                                     [ -B_MO_singlet,  -A_MO_singlet]])
    w, v = npl.eig(H_RPA_MO_singlet)
    wf, vf = npl.eig(H_RPA_MO_singlet_flip)
    # reminder: columns are eigenvectors
    # print(np.diag(np.dot(vf.T, np.dot(superoverlap, vf))))
    for i in range(len(w)):
        xt = v[:nov, i]
        xb = v[nov:, i]
        xpos = np.vstack((xt, xb))
        xneg = np.vstack((xb, xt))
        # print(np.dot(xpos.T, np.dot(H_RPA_MO_singlet, xpos)))
        # print(np.dot(xneg.T, np.dot(H_RPA_MO_singlet, xneg)))
        # print(np.dot(x.T, np.dot(superoverlap, x)))
    # ep = np.diag(np.dot(v.T, np.dot(H_RPA_MO_singlet, v)))
    # assert w == ep
