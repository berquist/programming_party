"""constants: Hold our handy-yet-complicated conversion factors."""

import math


amu2g = 1.6605402e-24
amu2kg = amu2g / 1000.0
bohr2ang = 0.529177249
bohr2m = bohr2ang * 1.0e-10
hartree2joule = 4.35974434e-18
planck = 6.6260755e-34
# planck = 6.62606957e-34
# pi = math.acos(-1.0)
pi = math.pi
planckbar = planck/(2*pi)
speed_of_light = 299792458
avogadro = 6.0221413e+23
rot_constant = planck/(8*pi*pi*speed_of_light)
vib_constant = math.sqrt((avogadro*hartree2joule*1000) /
                         (bohr2m*bohr2m)) / \
    (2*pi*speed_of_light*100)

# Or, we could use SymPy's units and extend them...
