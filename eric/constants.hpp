#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cmath>

const double masses[] = {
  0.000000, // Ghost
  1.007825, // H
  4.002603, // He
  6.015123, // Li
  9.012182, // Be
  10.012937, // B
  12.000000, // C
  14.003074, // N
  15.994915, // O
  18.998403, // F
  19.992440 // Ne
};

const double amu2kg = 1.6605402e-27;
const double bohr2m = 0.529177249e-10;
const double hartree2joule = 4.35974434e-18;
const double planck = 6.6260755e-34;
// const double planck = 6.62606957e-34;
const double pi = acos(-1.0);
const double planckbar = planck/(2*pi);
const double speed_of_light = 299792458;
const double avogadro = 6.0221413e+23;
const double rot_constant = planck/(8*pi*pi*speed_of_light);
const double vib_constant = sqrt((avogadro*hartree2joule*1000)/(bohr2m*bohr2m))/(2*pi*speed_of_light*100);

#endif /* CONSTANTS_HPP */
