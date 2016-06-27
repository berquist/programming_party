#!/usr/bin/env python

import numpy as np


def idx2(i, j):
    """Returns the compound index for two values.
    """

    if i > j:
        return (i*(i+1))/2 + j
    else:
        return (j*(j+1))/2 + i

def idx4(i, j, k, l):
    """Returns the compound index for four values.
    """

    ij = idx2(i, j)
    kl = idx2(k, l)
    return idx2(ij, kl)

def print_mat(mat):
    """Pretty-print a general NumPy matrix in a traditional format.
    """

    dim_rows, dim_cols = mat.shape
    # first, handle the column labels
    print(' ' * 5)
    for i in range(dim_cols):
        print('{0:12d}'.format(i+1), end='')
    print('')
    # then, handle the row labels
    for i in range(dim_rows):
        print('{0:5d}'.format(i+1), end='')
        # print the matrix data
        for j in range(dim_cols):
            print('{0:12.7f}'.format(mat[i][j]), end='')
        print('', end='\n')
    print('', end='\n')

    return

def np_load(filename):
    arr = np.load(filename)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Make the assumption that there's only a single array
        # present, even though *.npz files can hold multiple
        # arrays.
        arr = arr.items()[0][1]
    return arr


def read_arma_mat_ascii(armaasciifilename):
    """Given a file name, read it in as an ASCII-formatted Armadillo matrix.

    Currently, it supports matrices and cubes.

    The second line of the file contains the dimensions:
    rows, columns, slices (not sure about fields).

    Return a NumPy ndarray of shape [nslices, nrows, ncolumns].
    """

    with open(armaasciifilename) as armafile:
        next(armafile)
        shape = [int(x) for x in next(armafile).split()]

    if len(shape) == 2:
        rows, columns = shape
        slices = 0
    elif len(shape) == 3:
        rows, columns, slices = shape
    else:
        sys.exit(1)

    arma_mat = np.loadtxt(armaasciifilename, skiprows=2)

    if len(shape) == 2:
        arma_mat = arma_mat.ravel().reshape((rows, columns))
    elif len(shape) == 3:
        arma_mat = arma_mat.ravel().reshape((slices, rows, columns))
    else:
        sys.exit(1)

    return arma_mat

def matsym(amat, thrzer=1.0e-14):
    """This function returns
       1 if the matrix is symmetric to threshold THRZER
       2 if the matrix is antisymmetric to threshold THRZER
       3 if all elements are below THRZER
       0 otherwise (the matrix is unsymmetric about the diagonal)

    Copied from DALTON/gp/gphjj.F/MATSYM.
    thrzer taken from DALTON/include/thrzer.h
    """

    assert amat.shape[0] == amat.shape[1]

    n = amat.shape[0]

    isym = 1
    iasym = 2
    for j in range(n):
        # for i in range(j+1):
        # The +1 is so the diagonal elements are checked.
        for i in range(j+1):
            amats = abs(amat[i, j] + amat[j, i])
            amata = abs(amat[i, j] - amat[j, i])
            if amats > thrzer:
                iasym = 0
            if amata > thrzer:
                isym = 0

    return (isym + iasym)
