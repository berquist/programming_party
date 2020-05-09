import sys

import numpy as np


def idx2(i: int, j: int) -> int:
    """Returns the compound index for two values.
    """
    if i > j:
        return int((i * (i + 1)) / 2 + j)
    else:
        return int((j * (j + 1)) / 2 + i)


def idx4(i: int, j: int, k: int, l: int) -> int:
    """Returns the compound index for four values.
    """
    return idx2(idx2(i, j), idx2(k, l))


def print_mat(mat: np.ndarray) -> None:
    """Pretty-print a general NumPy matrix in a traditional format.
    """
    dim_rows, dim_cols = mat.shape
    # first, handle the column labels
    print(" " * 5)
    for i in range(dim_cols):
        print(f"{i + 1:12d}", end="")
    print("")
    # then, handle the row labels
    for i in range(dim_rows):
        print(f"{i + 1:5d}", end="")
        # print the matrix data
        for j in range(dim_cols):
            print(f"{mat[i][j]:12.7f}", end="")
        print("", end="\n")
    print("", end="\n")

    return


def np_load(filename: str) -> np.ndarray:
    arr = np.load(filename)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Make the assumption that there's only a single array
        # present, even though *.npz files can hold multiple
        # arrays.
        arr = list(arr.items())[0][1]
    return arr


def read_arma_mat_ascii(armaasciifilename: str) -> np.ndarray:
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


def matsym(amat: np.ndarray, thrzer: float = 1.0e-14) -> bool:
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
        # The +1 is so the diagonal elements are checked.
        for i in range(j + 1):
            amats = abs(amat[i, j] + amat[j, i])
            amata = abs(amat[i, j] - amat[j, i])
            if amats > thrzer:
                iasym = 0
            if amata > thrzer:
                isym = 0

    return bool(isym + iasym)
