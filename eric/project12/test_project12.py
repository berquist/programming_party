import numpy as np


def read_file(filename):
    iters = []
    energies_au = []
    energies_ev = []

    with open(filename) as fh:
        for _ in range(3):
            line = next(fh)
        for line in fh:
            tokens = line.split()
            itr = int(tokens[0])
            e_au = float(tokens[1])
            e_ev = float(tokens[2])
            iters.append(itr)
            energies_au.append(e_au)
            energies_ev.append(e_ev)

    assert len(iters) == len(energies_au) == len(energies_ev) == (iters[-1] + 1)
    return (np.array(energies_au), np.array(energies_ev))

if __name__ == '__main__':

    filename_cis = "h2o_sto3g_output_cis.txt"
    filename_rpa1 = "h2o_sto3g_output_rpa1.txt"
    filename_rpa2 = "h2o_sto3g_output_rpa2.txt"

    energies_au_cis, energies_ev_cis = read_file(filename_cis)
    energies_au_rpa1, energies_ev_rpa1 = read_file(filename_rpa1)
    energies_au_rpa2, energies_ev_rpa2 = read_file(filename_rpa2)

    # here we test the consistency of the references data

    # read from a file, this is nov = nocc * nvirt
    dim = 40
    assert energies_au_cis.shape[0] == dim
    assert energies_au_rpa1.shape[0] == 2 * dim
    assert energies_au_rpa2.shape[0] == dim

    # these should be all zeros
    print(energies_au_rpa1[dim:] - energies_au_rpa2)

    # these should all be zeros for these simple cases, but
    # deexcitation energies aren't necessarily the same as excitation
    # energies
    print(energies_au_rpa1[:dim][::-1] + energies_au_rpa2)
