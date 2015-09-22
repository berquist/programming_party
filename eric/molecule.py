#!/usr/bin/env python

"""molecule: Hold our representation of a Molecule and any helpful
external functions."""

from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.linalg as npl

import periodic_table as pt
import constants as c


class Molecule(object):
    """A Python representation of a molecule.
    """

    def __init__(self, filename):

        self.size = 0
        self.charges = []
        self.coords = []
        self.masses = []

        self.com = np.zeros(3)
        self.moi_tensor = np.zeros((3, 3))
        self.moi = np.zeros(3)

        self._load(filename)

        self._distances = np.zeros((self.size, self.size))
        self._calc_bonds()
        self._bond_angles = np.zeros((self.size, self.size, self.size))

        self._calc_com()
        self._calc_moi()

    def _load(self, filename):
        """Read in a pseudo-XYZ file for nuclear charges and coordinates.
        """

        handle = open(filename)
        rawfile = handle.readlines()
        self.size = int(float(rawfile[0]))
        self.charges = np.zeros(self.size, 'int')
        self.coords = np.zeros((self.size, 3), 'float')
        for idx, line in enumerate(rawfile[1:]):
            l = line.split()
            self.charges[idx] = int(float(l[0]))
            self.coords[idx] = map(float, l[1:])
        handle.close()

        # with open(filename) as handle:

        self.masses = np.array([pt.Mass[pt.Element[charge]]
                                for charge in self.charges])

        return

    def _build_distance_matrix(self):
        """Build the distance matrix (bond distances) from the coordinates.
        """

        for i in range(0, self.size):
            for j in range(i+1, self.size):
                self._distances[i][j] = npl.norm(self.coords[i] - self.coords[j])
                self._distances[j][i] = self._distances[i][j]

        return

    def _calc_bond(self, i, j):
        """Calculate the bond length between atoms with indices i and j.
        """
        return self._distances[i][j]
        # return npl.norm(self.coords[i] - self.coords[j])

    def _calc_angle(self, i, j, k):
        """Calculate the angle between atoms with indices i, j, and k.  The
        atom with index j is central atom.
        """

        return -1.0

    def _calc_angle_oop(self, i, j, k, l):
        """Calculate the out-of-plane angle between atoms with indices i, j,
        k, and l.
        """

        return -1.0

    def _calc_angle_torsion(self, i, j, k, l):
        """Calculate the torsion angle between atoms with indices i, j, k,
        and l.
        """

        return -1.0

    def _calc_bonds(self):
        """Calculate all possible atom distances and store them.
        """

        self._build_distance_matrix()

    def _calc_angles(self):
        """Calculate all possible bond angles and store them.
        """

        pass

    def _calc_oop_angles(self):
        """Calculate all possible out-of-plane angles and store them.
        """

        pass

    def _calc_torsion_angles(self):
        """Calculate all possible torsional angles and store them.
        """

        pass

    def _calc_com(self, translatep=True):
        """Calculate the center of mass and store it.
        """

        # Use vectorized operations rather than explicit loops.
        mass_sum = np.sum(self.masses)
        CMx = np.sum(self.coords[:, 0] * self.masses) / mass_sum
        CMy = np.sum(self.coords[:, 1] * self.masses) / mass_sum
        CMz = np.sum(self.coords[:, 2] * self.masses) / mass_sum
        # An example of simultaneous assignment.
        self.com[0], self.com[1], self.com[2] = CMx, CMy, CMz
        # We could also do this:
        # self.com = [CMx, CMy, CMz]

        # Should we translate this molecule to its center of mass?
        if translatep:
            self.translate(-CMx, -CMy, -CMz)

        return

    def _calc_moi(self):
        """Calculate the moment of inertia tensor, then diagonalize it to
        obtain the 3 principal moments of inertia.
        """

        # this is the "traditional" way of looping explicitly over all atoms
        # for i in range(self.size):
        #     mi = masses[i]
        #     xi = self.coords[i][0]
        #     yi = self.coords[i][1]
        #     zi = self.coords[i][2]
        #     # diagonal terms
        #     self.moi_tensor[0][0] += (mi * ((yi*yi) + (zi*zi)))
        #     self.moi_tensor[1][1] += (mi * ((xi*xi) + (zi*zi)))
        #     self.moi_tensor[2][2] += (mi * ((xi*xi) + (yi*yi)))
        #     # off-diagonal terms
        #     self.moi_tensor[0][1] += (mi * xi * yi)
        #     self.moi_tensor[0][2] += (mi * xi * zi)
        #     self.moi_tensor[1][2] += (mi * yi * zi)

        # this is the "vectorized" way doing the above implicitly
        # diagonal terms
        # pylint: disable=line-too-long
        self.moi_tensor[0][0] = np.sum(self.masses * (self.coords[:, 1]**2 + self.coords[:, 2]**2))
        self.moi_tensor[1][1] = np.sum(self.masses * (self.coords[:, 0]**2 + self.coords[:, 2]**2))
        self.moi_tensor[2][2] = np.sum(self.masses * (self.coords[:, 0]**2 + self.coords[:, 1]**2))
        # off-diagonal terms
        self.moi_tensor[0][1] = np.sum(self.masses * self.coords[:, 0] * self.coords[:, 1])
        self.moi_tensor[0][2] = np.sum(self.masses * self.coords[:, 0] * self.coords[:, 2])
        self.moi_tensor[1][2] = np.sum(self.masses * self.coords[:, 1] * self.coords[:, 2])

        self.moi_tensor[1][0] = self.moi_tensor[0][1]
        self.moi_tensor[2][0] = self.moi_tensor[0][2]
        self.moi_tensor[2][1] = self.moi_tensor[1][2]

        # diagonalize the MOI tensor to get the principal moments
        self.moi = npl.eigvalsh(self.moi_tensor)

        return

    def _calc_rot_const(self):
        """Compute the rotational constants in 1/cm and MHz.
        """

        pass

    def _print_rotor_type(self):
        """Based on the principal moments of inertia, print a description of
        its rotor type.
        """

        # a little unpacking trick
        A, B, C = self.moi
        thresh = 1.0e-4
        if self.size == 2:
            print('The molecule is diatomic.')
        elif A < thresh:
            print('The molecule is linear.')
        elif (abs(A-B) < thresh) and (abs(B-C) < thresh):
            print('The molecule is a spherical top.')
        elif (abs(A-B) < thresh) and (abs(B-C) > thresh):
            print('The molecule is an oblate symmetric top.')
        elif (abs(A-B) > thresh) and (abs(B-C) < thresh):
            print('The molecule is a prolate symmetric top.')
        else:
            print('The molecule is an asymmetric top.')

    def print_geom(self):
        """Print this molecule's geometry in bohr (just regurgitate the XYZ
        file).
        """

        pass

    def print_bonds(self):
        """Print this molecule's bonds.
        """

        print('Interatomic distances [bohr]:')
        s = '{:2d} {:2d} {:8.5f}'.format
        for i in range(self.size):
            for j in range(i):
                print(s(i, j, self.bond(i, j)))

        return

    def print_angles(self):
        """Print this molecule's angles.
        """

        print('Angles [degrees]:')
        s = '{:2d} {:2d} {:2d} {:10.6f}'.format
        for i in range(self.size):
            for j in range(i):
                for k in range(j):
                    if (self.bond(i, j) < 4.0) and (self.bond(j, k) < 4.0):
                        print(s(i, j, k, self.angle(i, j, k)))

        return

    def print_oop_angles(self):
        """Print this molecule's out-of-plane angles.
        """

        print('Out-of-plane angles [degrees]:')
        s = '{:2d} {:2d} {:2d} {:2d} {:10.6f}'.format
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        # pylint: disable=line-too-long
                        if (i != j) and (i != k) and (i != l) and (j != k) and (j != l) and (k != l) \
                           and (self.bond(i, k) < 4.0) and (self.bond(k, j) < 4.0) and (self.bond(k, l) < 4.0):
                            print(s(i, j, k, l, self.angle_oop(i, j, k, l)))

        return

    def print_torsion_angles(self):
        """Print this molecule's torsion angles.
        """

        print('Torsion/dihedral angles [degrees]:')
        s = '{:2d} {:2d} {:2d} {:2d} {:10.6f}'.format
        for i in range(self.size):
            for j in range(i):
                for k in range(j):
                    for l in range(k):
                        # pylint: disable=line-too-long
                        if (self.bond(i, j) < 4.0) and (self.bond(j, k) < 4.0) and (self.bond(k, l) < 4.0):
                            print(s(i, j, k, l, self.angle_torsion(i, j, k, l)))

        return

    def print_com(self):
        """Print this molecule's center of mass.
        """
        # pylint: disable=line-too-long
        print('Molecular center of mass [bohr]: {:12.8f} {:12.8f} {:12.8f}'.format(*self.com))

        return

    def print_moi(self):
        """Print the moment of inertia tensor and the principal moments of
        inertia.
        """

        print('Moment of inertia tensor [amu][bohr]^2:')
        print(self.moi_tensor)
        print('Principal moments of inertia:')
        print('    [amu][bohr]^2: {:16.8f} {:16.8f} {:16.8f}'.format(*self.moi))
        conv = c.bohr2ang * c.bohr2ang
        print('[amu][angstrom]^2: {:16.8f} {:16.8f} {:16.8f}'.format(*(self.moi * conv)))
        conv = c.amu2g * c.bohr2ang * 1e-8 * c.bohr2ang * 1e-8
        print('        [g][cm]^2: {:16.8e} {:16.8e} {:16.8e}'.format(*(self.moi * conv)))
        self._print_rotor_type()

        return

    def print_rot_const(self):
        """Print the rotational constants, in MHz and 1/cm.
        """

        pass

    def rotate(self, phi):
        """Rotate the coordinates of the molecule by phi degrees.
        """

        pass

    def translate(self, x, y, z):
        """Translate (shift) the coordinates of the molecule in the 3
        Cartesian directions.
        """

        # loop style
        # for i in range(self.size):
        #     self.coords[i][0] += x
        #     self.coords[i][1] += y
        #     self.coords[i][2] += z

        # vectorized style
        self.coords[:, 0] += x
        self.coords[:, 1] += y
        self.coords[:, 2] += z

        return

    def bond(self, i, j):
        """Return the distance between atoms i and j in bohr.
        """

        return self._distances[i][j]

    def angle(self, i, j, k):
        """Return the angle between atoms i, j, and k in degrees.  Atom j is
        the central atom.
        """

        # Calculate the angle on the fly. Don't use any stored values.
        return self._calc_angle(i, j, k)

    def angle_torsion(self, i, j, k, l):
        """Return the torsion angle between atoms i, j, k, and l.
        """

        # Calculate the angle on the fly. Don't use any stored values.
        return self._calc_angle_torsion(i, j, k, l)

    def angle_oop(self, i, j, k, l):
        """Return the out-of-plane angle between atoms i, j, k, and l.
        """

        # Calculate the angle on the fly. Don't use any stored values.
        return self._calc_angle_oop(i, j, k, l)
