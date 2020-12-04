import numpy as np

from eric.utils import np_load


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--nelec', type=int, default=10)

    args = parser.parse_args()

    return args


def ao_mo_4index_noddy(TEI_AO, C):

    nbasis, norb = C.shape

    TEI_MO = np.zeros(shape=TEI_AO.shape)

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    for mu in range(nbasis):
                        for nu in range(nbasis):
                            for lm in range(nbasis):
                                for sg in range(nbasis):
                                    TEI_MO[p, q, r, s] += \
                                        C[mu, p] * \
                                        C[nu, q] * \
                                        C[lm, r] * \
                                        C[sg, s] * \
                                        TEI_AO[mu, nu, lm, sg]

    return TEI_MO


def ao_mo_4index_smart_quarter(TEI_AO, C):

    nbasis, norb = C.shape

    TEI_MO = np.zeros(shape=TEI_AO.shape)

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

    return TEI_MO


def ao_mo_full_einsum(TEI_AO, C):

    return np.einsum('mnls,ma,nb,lc,sd->abcd', TEI_AO, C, C, C, C)


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




if __name__ == "__main__":

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2

    # Load in the previously calculated quantities we need from a
    # converged SCF run.
    TEI_AO = np_load('TEI_AO.npz')
    C = np_load('C.npz')
    F_MO = np_load('F_MO.npz')

    # Peform the full 4-index AO->MO transformation.
    # ao_mo_4index_noddy(TEI_MO, TEI_AO, C)
    TEI_MO = ao_mo_4index_smart_quarter(TEI_AO, C)
    # Save them MO-basis 2-electron spatial orbitals to disk for later
    # use.
    np.savez_compressed('TEI_MO.npz', TEI_MO)

    # The MO energies are the diagonal elements of the Fock matrix in
    # the MO basis.
    E = np.diag(F_MO)

    # Calculate the MP2 energy.
    E_MP2 = calc_mp2_energy(TEI_MO, E, nocc)

    print('E_MP2: {:20.12f}'.format(E_MP2))
