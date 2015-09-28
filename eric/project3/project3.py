#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np

from ..utils import print_mat

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

nelec = 10
dim = 7

stub = "h2o_sto3g_"

filename_enuc = stub + "enuc.dat"
filename_s = stub + "s.dat"
filename_t = stub + "t.dat"
filename_v = stub + "v.dat"
filename_eri = stub + "eri.dat"

enuc = parse_file_1(filename_enuc)
mat_s = parse_int_file_2(filename_s, dim)
mat_t = parse_int_file_2(filename_t, dim)
mat_v = parse_int_file_2(filename_v, dim)
mat_eri = parse_int_file_4(filename_eri, dim)

print("Nuclear repulsion energy = {}\n".format(enuc))
print("Overlap Integrals:")
print_mat(mat_s)
print("Kinetic-Energy Integrals:")
print_mat(mat_t)
print("Nuclear Attraction Integrals:")
print_mat(mat_v)
