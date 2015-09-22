#!/usr/bin/env python

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
        print('{0:11d}'.format(i+1), end='')
    print('')
    # then, handle the row labels
    for i in range(dim_rows):
        print('{0:5d}'.format(i+1), end='')
        # print the matrix data
        for j in range(dim_cols):
            print('{0:11.7f}'.format(mat[i][j]), end='')
        print('', end='\n')
    print('', end='\n')
