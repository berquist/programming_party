import numpy as np

np.set_printoptions(precision=8, linewidth=200, suppress=True)

from eric.utils import np_load


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--nelec", type=int, default=10)

    args = parser.parse_args()

    return args


def mo_so_4index(TEI_SO, TEI_MO):

    norb = TEI_MO.shape[0]

    for p in range(2 * norb):
        for q in range(2 * norb):
            for r in range(2 * norb):
                for s in range(2 * norb):
                    lint = (
                        TEI_MO[p // 2, r // 2, q // 2, s // 2]
                        * (p % 2 == r % 2)
                        * (q % 2 == s % 2)
                    )
                    rint = (
                        TEI_MO[p // 2, s // 2, q // 2, r // 2]
                        * (p % 2 == s % 2)
                        * (q % 2 == r % 2)
                    )
                    TEI_SO[p, q, r, s] = lint - rint

    return


def make_amplitude_guess(TEI_SO, E, nsocc):

    nsorb = TEI_SO.shape[0]

    T1 = np.zeros(shape=(nsorb, nsorb))
    T2 = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    T2[i, j, a, b] += TEI_SO[i, j, a, b] / (
                        E[i // 2] + E[j // 2] - E[a // 2] - E[b // 2]
                    )

    return T1, T2


def calc_mp2_energy(TEI_SO, T2, nsocc):

    nsorb = TEI_SO.shape[0]

    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)

    return np.einsum("ijab,ijab->", TEI_SO[o, o, v, v], T2[o, o, v, v]) / 4.0


def build_fock_spin_orbital(TEI_SO, H, nsocc):

    nsorb = TEI_SO.shape[0]

    F_SO = np.zeros(shape=(nsorb, nsorb))

    for p in range(nsorb):
        for q in range(nsorb):
            F_SO[p, q] += H[p // 2, q // 2]
            for m in range(nsocc):
                F_SO[p, q] += TEI_SO[p, m, q, m]

    return F_SO


def build_diagonal_2(F_SO, nsocc):
    """Build the 2-index energy matrix."""

    nsorb = F_SO.shape[0]

    D = np.zeros(shape=(nsorb, nsorb))

    for i in range(0, nsocc):
        for a in range(nsocc, nsorb):
            D[i, a] += F_SO[i, i] - F_SO[a, a]

    return D


def build_diagonal_4(F_SO, nsocc):
    """Build the 4-index energy matrix."""

    nsorb = F_SO.shape[0]

    D = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    for i in range(0, nsocc):
        for j in range(0, nsocc):
            for a in range(nsocc, nsorb):
                for b in range(nsocc, nsorb):
                    D[i, j, a, b] += F_SO[i, i] + F_SO[j, j] - F_SO[a, a] - F_SO[b, b]

    return D


def build_intermediates_2index_take2(TEI_SO, F_SO, T1, tau_twiddle, nsocc):
    nsorb = TEI_SO.shape[0]

    F = np.zeros(shape=(nsorb, nsorb))

    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)
    nsvir = nsorb - nsocc

    # equation (3)
    F[v, v] += (1 - np.eye(nsvir)) * F_SO[v, v]
    # print(F)
    F[v, v] -= 0.5 * np.einsum("me,ma->ae", F_SO[o, v], T1[o, v])
    F[v, v] += np.einsum("mf,mafe->ae", T1[o, v], TEI_SO[o, v, v, v])
    F[v, v] -= 0.5 * np.einsum(
        "mnaf,mnef->ae", tau_twiddle[o, o, v, v], TEI_SO[o, o, v, v]
    )

    # equation (4)
    F[o, o] += (1 - np.eye(nsocc)) * F_SO[o, o]
    F[o, o] += 0.5 * np.einsum("ie,me->mi", T1[o, v], F_SO[o, v])
    F[o, o] += np.einsum("ne,mnie->mi", T1[o, v], TEI_SO[o, o, o, v])
    F[o, o] += 0.5 * np.einsum(
        "inef,mnef->mi", tau_twiddle[o, o, v, v], TEI_SO[o, o, v, v]
    )

    # equation (5)
    F[o, v] = F_SO[o, v] + np.einsum("nf,mnef->me", T1[o, v], TEI_SO[o, o, v, v])

    # TODO where is the virt-occ block?
    # F[v, o] = F[o, v].T

    return F


def form_tau(t1, t2, nsocc):
    nsorb = t1.shape[0]
    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)
    tau = t2.copy()
    tmp = np.einsum("ia,jb->ijab", t1[o, v], t1[o, v])
    tau[o, o, v, v] += tmp
    # ib,ja->ijab
    tau[o, o, v, v] -= tmp.swapaxes(2, 3)
    return tau


def form_tau_twiddle(t1, t2, nsocc):
    nsorb = t1.shape[0]
    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)
    tau = t2.copy()
    tmp = 0.5 * np.einsum("ia,jb->ijab", t1[o, v], t1[o, v])
    tau[o, o, v, v] += tmp
    # ib,ja->ijab
    tau[o, o, v, v] -= tmp.swapaxes(2, 3)
    return tau


def build_intermediates_4index_take2(TEI_SO, F_SO, T1, T2, tau, nsocc):
    """Form the 4-index intermediate W using equations 6-8."""

    nsorb = TEI_SO.shape[0]

    W = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)

    # equation (6)
    W[o, o, o, o] = TEI_SO[o, o, o, o]
    # permutation term
    perm = np.einsum("je,mnie->mnij", T1[o, v], TEI_SO[o, o, o, v])
    W[o, o, o, o] += perm
    # ie,mnje->mnij
    W[o, o, o, o] -= perm.swapaxes(2, 3)
    W[o, o, o, o] += 0.25 * np.einsum(
        "ijef,mnef->mnij", tau[o, o, v, v], TEI_SO[o, o, v, v]
    )

    # equation (7)
    W[v, v, v, v] = TEI_SO[v, v, v, v]
    # permutation term
    perm = np.einsum("mb,amef->abef", T1[o, v], TEI_SO[v, o, v, v])
    W[v, v, v, v] -= perm
    # ma,bmef->abef
    W[v, v, v, v] += perm.swapaxes(0, 1)
    W[v, v, v, v] += 0.25 * np.einsum(
        "mnab,mnef->abef", tau[o, o, v, v], TEI_SO[o, o, v, v]
    )

    # equation (8)
    W[o, v, v, o] = TEI_SO[o, v, v, o]
    W[o, v, v, o] += np.einsum("jf,mbef->mbej", T1[o, v], TEI_SO[o, v, v, v])
    W[o, v, v, o] -= np.einsum("nb,mnej->mbej", T1[o, v], TEI_SO[o, o, v, o])
    # 0.5 * T2_{jn,fb} + T1_{j,f}T1_{n,b}
    # 0.5 * T2[o, o, v, v] + T1[o, v]T1[o, v]
    # be lazy and expand the whole thing
    # contract over `nf` in both cases
    W[o, v, v, o] -= 0.5 * np.einsum(
        "jnfb,mnef->mbej", T2[o, o, v, v], TEI_SO[o, o, v, v]
    )
    W[o, v, v, o] -= np.einsum(
        "jf,nb,mnef->mbej", T1[o, v], T1[o, v], TEI_SO[o, o, v, v]
    )

    return W


def update_amplitudes_T1_take2(TEI_SO, F_SO, T1, T2, nsocc, D1, F):
    """Update the T1 amplitudes (equation 1)."""
    nsorb = F_SO.shape[0]
    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)

    uT1 = np.zeros(shape=(nsorb, nsorb))

    uT1[o, v] = F_SO[o, v]
    uT1[o, v] += np.einsum("ie,ae->ia", T1[o, v], F[v, v])
    uT1[o, v] -= np.einsum("ma,mi->ia", T1[o, v], F[o, o])
    uT1[o, v] += np.einsum("imae,me->ia", T2[o, o, v, v], F[o, v])
    uT1[o, v] -= np.einsum("nf,naif->ia", T1[o, v], TEI_SO[o, v, o, v])
    uT1[o, v] -= 0.5 * np.einsum("imef,maef->ia", T2[o, o, v, v], TEI_SO[o, v, v, v])
    uT1[o, v] -= 0.5 * np.einsum("mnae,nmei->ia", T2[o, o, v, v], TEI_SO[o, o, v, o])
    uT1[o, v] /= D1[o, v]

    return uT1


def update_amplitudes_T2_take2(TEI_SO, F, T1, T2, nsocc, D2, W, tau):
    """Update the T2 amplitudes (equation 2)."""
    nsorb = F.shape[0]
    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)

    uT2 = np.zeros(shape=(nsorb, nsorb, nsorb, nsorb))

    uT2[o, o, v, v] = TEI_SO[o, o, v, v]

    # term 2
    # be lazy and expand for now
    left = np.einsum("ijae,be->ijab", T2[o, o, v, v], F[v, v])
    right = 0.5 * np.einsum("ijae,mb,me->ijab", T2[o, o, v, v], T1[o, v], F[o, v])
    res = left - right
    # permute `ab`
    uT2[o, o, v, v] += res - res.swapaxes(2, 3)

    # term 3
    # be lazy and expand for now
    left = np.einsum("imab,mj->ijab", T2[o, o, v, v], F[o, o])
    right = 0.5 * np.einsum("imab,je,me->ijab", T2[o, o, v, v], T1[o, v], F[o, v])
    res = left - right
    # permute `ij`
    uT2[o, o, v, v] -= res + res.swapaxes(0, 1)

    # term 4
    uT2[o, o, v, v] += 0.5 * np.einsum(
        "mnab,mnij->ijab", tau[o, o, v, v], W[o, o, o, o]
    )
    # term 5
    uT2[o, o, v, v] += 0.5 * np.einsum(
        "ijef,abef->ijab", tau[o, o, v, v], W[v, v, v, v]
    )
    # term 6
    # (1 - p_{ab})⋅(1 - p_{ij}) => p_{ab}⋅p_{ij} - p_{ab} - p_{ij} + 1
    left = np.einsum("imae,mbej->ijab", T2[o, o, v, v], W[o, v, v, o])
    right = np.einsum("ie,ma,mbej->ijab", T1[o, v], T1[o, v], TEI_SO[o, v, v, o])
    res = left - right
    uT2[o, o, v, v] += res
    uT2[o, o, v, v] -= res.swapaxes(0, 1)
    uT2[o, o, v, v] -= res.swapaxes(2, 3)
    uT2[o, o, v, v] += res.swapaxes(0, 1).swapaxes(2, 3)
    # term 7
    normal = np.einsum("ie,abej->ijab", T1[o, v], TEI_SO[v, v, v, o])
    # je,abei->ijab
    perm = normal.swapaxes(0, 1)
    uT2[o, o, v, v] += normal
    uT2[o, o, v, v] -= perm
    # term 8
    normal = np.einsum("ma,mbij->ijab", T1[o, v], TEI_SO[o, v, o, o])
    # mb,maij->ijab
    perm = normal.swapaxes(2, 3)
    uT2[o, o, v, v] -= normal
    uT2[o, o, v, v] += perm

    uT2[o, o, v, v] /= D2[o, o, v, v]

    return uT2


def calc_ccsd_energy_take2(TEI_SO, F_SO, T1, T2, nsocc):
    nsorb = F_SO.shape[0]
    o = slice(0, nsocc)
    v = slice(nsocc, nsorb)
    return (
        np.einsum("ia,ia->", F_SO[o, v], T1[o, v])
        + 0.25 * np.einsum("ijab,ijab->", TEI_SO[o, o, v, v], T2[o, o, v, v])
        + 0.5 * np.einsum("ijab,ia,jb->", TEI_SO[o, o, v, v], T1[o, v], T1[o, v])
    )


def main():

    args = getargs()

    nelec = args.nelec
    nocc = nelec // 2
    nsocc = nocc * 2

    H = np_load("H.npz")
    E = np.diag(np_load("F_MO.npz"))
    TEI_MO = np_load("TEI_MO.npz")

    TEI_SO = np.zeros(shape=np.array(TEI_MO.shape) * 2)

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

    print("E(MP2): {:20.12f}".format(E_MP2))
    assert abs(E_MP2 - -0.049149636120) < 1.0e-13

    # Form the effective two-particle excitation operators.
    tau = form_tau(T1, T2, nsocc)
    tau_twiddle = form_tau_twiddle(T1, T2, nsocc)

    # Form the one- and two-particle intermediates.
    F = build_intermediates_2index_take2(TEI_SO, F_SO, T1, tau_twiddle, nsocc)
    # print(F)
    W = build_intermediates_4index_take2(TEI_SO, F_SO, T1, T2, tau, nsocc)

    # Calculate the first updated set of amplitudes.
    T1 = update_amplitudes_T1_take2(TEI_SO, F_SO, T1, T2, nsocc, D1, F)
    T2 = update_amplitudes_T2_take2(TEI_SO, F, T1, T2, nsocc, D2, W, tau)

    E_CCSD = calc_ccsd_energy_take2(TEI_SO, F_SO, T1, T2, nsocc)

    print("Iter E(CCSD) sum(T1) sum(T2)")
    t = " {:4d} {:20.12f} {} {}".format
    print(t(0, E_CCSD, np.sum(T1), np.sum(T2)))

    thresh_e = 1.0e-15
    iteration = 1
    max_iterations = 20

    while iteration < max_iterations:

        tau = form_tau(T1, T2, nsocc)
        tau_twiddle = form_tau_twiddle(T1, T2, nsocc)

        F = build_intermediates_2index_take2(TEI_SO, F_SO, T1, tau_twiddle, nsocc)
        W = build_intermediates_4index_take2(TEI_SO, F_SO, T1, T2, tau, nsocc)

        T1 = update_amplitudes_T1_take2(TEI_SO, F_SO, T1, T2, nsocc, D1, F)
        T2 = update_amplitudes_T2_take2(TEI_SO, F, T1, T2, nsocc, D2, W, tau)

        E_CCSD = calc_ccsd_energy_take2(TEI_SO, F_SO, T1, T2, nsocc)

        print(t(iteration, E_CCSD, np.sum(T1), np.sum(T2)))

        iteration += 1

    return


if __name__ == "__main__":
    main()
