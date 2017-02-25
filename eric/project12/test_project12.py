import pytest

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
    # return (np.array(energies_au), np.array(energies_ev))
    return np.array(energies_au)


energies_params = ['h2o_sto3g', 'h2o_dz', 'h2o_dzp', 'ch4_sto3g']
@pytest.fixture(params=energies_params)
def energies_from_file(request):
    filename_cis = '{}_output_cis.txt'.format(request.param)
    filename_rpa1 = '{}_output_rpa1.txt'.format(request.param)
    filename_rpa2 = '{}_output_rpa2.txt'.format(request.param)

    energies_au_cis = read_file(filename_cis)
    energies_au_rpa1 = read_file(filename_rpa1)
    energies_au_rpa2 = read_file(filename_rpa2)

    return energies_au_cis, energies_au_rpa1, energies_au_rpa2


def test_dim(energies_from_file):
    energies_au_cis, energies_au_rpa1, energies_au_rpa2 = energies_from_file

    dim = energies_au_cis.shape[0]
    assert energies_au_rpa1.shape[0] == 2 * dim
    assert energies_au_rpa2.shape[0] == dim


def test_zeros_excitation(energies_from_file):
    energies_au_cis, energies_au_rpa1, energies_au_rpa2 = energies_from_file
    dim = energies_au_cis.shape[0]

    # These shoudl all be zeros in all cases.
    should_be_zeros = energies_au_rpa1[dim:] - energies_au_rpa2
    assert np.zeros(dim).all() == should_be_zeros.all()


def test_zeros_deexcitation(energies_from_file):
    energies_au_cis, energies_au_rpa1, energies_au_rpa2 = energies_from_file
    dim = energies_au_cis.shape[0]

    # These should all be zeros for these simple cases, but
    # deexcitation energies aren't necessarily the same as excitation
    # energies, right?
    should_be_zeros = energies_au_rpa1[:dim][::-1] + energies_au_rpa2
    assert np.zeros(dim).all() == should_be_zeros.all()



if __name__ == '__main__':

    pass

