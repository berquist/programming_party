#include <cstdio>
#include <cmath>
#include "molecule.hpp"
#include "utils.hpp"
#include "constants.hpp"

void Molecule::print_geom()
{
  for (int i = 0; i < natom; i++)
    printf("%3.0f %8.5f %8.5f %8.5f\n", zvals[i], geom[i][0], geom[i][1], geom[i][2]);
}

void Molecule::print_bonds()
{
  printf("\nInteratomic distances [bohr]:\n");
  for (int i = 0; i < natom; i++)
    for (int j = 0; j < i; j++)
      printf("%3d %3d %10.6f\n", i, j, bond(i,j));
}

void Molecule::print_angles()
{
  printf("\nAngles [degrees]:\n");
  for (int i = 0; i < natom; i++)
    for (int j = 0; j < i; j++)
      for (int k = 0; k < j; k++)
	printf("%3d %3d %3d %10.6f\n", i, j, k, angle(i,j,k));
}

void Molecule::print_oop_angles()
{
  printf("\nOut-of-plane angles [degrees]:\n");
  for (int i = 0; i < natom; i++)
    for (int k = 0; k < natom; k++)
      for (int j = 0; j < natom; j++)
	for (int l = 0; l < j; l++)
	  if (i!=j && i!=k && i!=l && j!=k && k!=l && bond(i,k) < 4.0 && bond(k,j) < 4.0 && bond(k,l) < 4.0)
	    printf("%3d %3d %3d %3d %10.6f\n", i, j, k, l, angle_oop(i,j,k,l));
}

void Molecule::print_torsion_angles()
{
  printf("\nTorsion/Dihedral angles [degrees]:\n");
  for (int i = 0; i < natom; i++)
    for (int j = 0; j < i; j++)
      for (int k = 0; k < j; k++)
	for (int l = 0; l < k; l++)
	  if (bond(i,j) < 4.0 && bond(j,k) < 4.0 && bond(k,l) < 4.0)
	    printf("%3d %3d %3d %3d %10.6f\n", i, j, k, l, angle_torsion(i,j,k,l));
}

void Molecule::print_com()
{
  calc_com(true);
}

void Molecule::print_moi()
{
  calc_moi();
}

void Molecule::print_rot_const()
{
  calc_rot_const();
}

void Molecule::rotate(double phi)
{
  
}

void Molecule::translate(double x, double y, double z)
{
  for (int i = 0; i < natom; i++) {
    geom[i][0] += x;
    geom[i][1] += y;
    geom[i][2] += z;
  }
}

// Returns the distance between atoms i and j in bohr.
double Molecule::bond(int i, int j)
{
  return calc_bond(i,j);
}

// Returns the value of the unit vector between atoms i and j
// in the cart direction (cart=0=x, cart=1=y, cart=2=z)
double Molecule::calc_unit(int i, int j, int cart)
{
  return -(geom[i][cart] - geom[j][cart]) / bond(i,j);
}

// Returns the angle between atoms i, j, and k in degrees.
// Atom j is the central atom.
double Molecule::angle(int i, int j, int k)
{
  return calc_angle(i,j,k);
}

//
double Molecule::angle_oop(int i, int j, int k, int l)
{
  return calc_angle_oop(i,j,k,l);
}

//
double Molecule::angle_torsion(int i, int j, int k, int l)
{
  return calc_angle_torsion(i,j,k,l);
}

Molecule::Molecule(int n, int q)
{
  natom = n;
  charge = q;
  zvals = new double[natom];
  geom = new double* [natom];
  for (int i = 0; i < natom; i++) {
    geom[i] = new double[3];
  }
  moi = new double* [3];
  moi_abc = new double[3];
  for (int i = 0; i < 3; i++) {
    moi[i] = new double[3];
  }
  for (int i = 0; i < 3; i++) {
    moi_abc[i] = 0.0;
    for (int j = 0; j < 3; j++) {
      moi[i][j] = 0.0;
    }
  }
}

Molecule::~Molecule()
{
  delete[] zvals;
  for (int i = 0; i < natom; i++) {
    delete[] geom[i];
  }
  delete[] geom;
  for (int i = 0; i < 3; i++) {
    delete[] moi[i];
  }
  delete[] moi;
  delete[] moi_abc;
}

double Molecule::calc_bond(int i, int j)
{
  double dx = geom[i][0] - geom[j][0];
  double dy = geom[i][1] - geom[j][1];
  double dz = geom[i][2] - geom[j][2];
  return sqrt((dx*dx) + (dy*dy) + (dz*dz));
}

// Calculates the angle between atoms i, j, and k in degrees.
// Atom j is the central atom.
double Molecule::calc_angle(int i, int j, int k)
{
  // return acos((calc_unit(i, j, 0) * calc_unit(j, k, 0)) + 
  // 	      (calc_unit(i, j, 1) * calc_unit(j, k, 1)) + 
  // 	      (calc_unit(i, j, 2) * calc_unit(j, k, 2))) * (180.0/acos(-1.0));
  return acos((calc_unit(j,i,0) * calc_unit(j,k,0)) + 
	      (calc_unit(j,i,1) * calc_unit(j,k,1)) + 
	      (calc_unit(j,i,2) * calc_unit(j,k,2))) * (180.0/acos(-1.0));
}

double Molecule::calc_angle_oop(int i, int j, int k, int l)
{
  // \mathrm{sin } \theta_{ijkl} = \frac{\mathbf{\tilde{e}_{kj}} \times \mathbf{\tilde{e}_{kl}}}{\mathrm{sin } \phi_{jkl}} \cdot \mathbf{\tilde{e}_{ki}}

  // double ekj[3] = {calc_unit(k, j, 0), calc_unit(k, j, 1), calc_unit(k, j, 2)};
  // double ekl[3] = {calc_unit(k, l, 0), calc_unit(k, l, 1), calc_unit(k, l, 2)};
  // double eki[3] = {calc_unit(k, i, 0), calc_unit(k, i, 1), calc_unit(k, i, 2)};

  // double ekj_cross_kl[3];
  // ekj_cross_kl[0] = (ekj[1]*ekl[2]) - (ekj[2]*ekl[1]);
  // ekj_cross_kl[1] = (ekj[2]*ekl[0]) - (ekj[0]*ekl[2]);
  // ekj_cross_kl[2] = (ekj[0]*ekl[1]) - (ekj[1]*ekl[0]);

  // double ekj_cross_kl_dot_ki[3];
  // ekj_cross_kl_dot_ki[0] = ekj_cross_kl[0] * eki[0];
  // ekj_cross_kl_dot_ki[1] = ekj_cross_kl[1] * eki[1];
  // ekj_cross_kl_dot_ki[2] = ekj_cross_kl[2] * eki[2];

  // double theta = (ekj_cross_kl_dot_ki[0] + ekj_cross_kl_dot_ki[1] + ekj_cross_kl_dot_ki[2])/sin(angle(j,k,l));

  // if (theta < -1.0) theta = asin(-1.0);
  // else if (theta > 1.0) theta = asin(1.0);
  // else theta = asin(theta);

  double ejkl_x = (calc_unit(k,j,1) * calc_unit(k,l,2)) - (calc_unit(k,j,2) * calc_unit(k,l,1));
  double ejkl_y = (calc_unit(k,j,2) * calc_unit(k,l,0)) - (calc_unit(k,j,0) * calc_unit(k,l,2));
  double ejkl_z = (calc_unit(k,j,0) * calc_unit(k,l,1)) - (calc_unit(k,j,1) * calc_unit(k,l,0));
  
  double exx = ejkl_x * calc_unit(k,i,0);
  double eyy = ejkl_y * calc_unit(k,i,1);
  double ezz = ejkl_z * calc_unit(k,i,2);

  double theta = (exx + eyy + ezz)/sin(angle(j,k,l));

  if (theta < -1.0) theta = asin(-1.0);
  else if (theta > 1.0) theta = asin(1.0);
  else theta = asin(theta);

  return theta * (180.0/acos(-1.0));
}

double Molecule::calc_angle_torsion(int i, int j, int k, int l)
{
  return 0.0;
}

void Molecule::calc_bonds()
{

}

void Molecule::calc_angles()
{
  // // calculate the unit vector between each atom
  // for (int i = 0; i < natom; i++) {
  //   ex[i][i] = ey[i][i] = ez[i][i] = 0.0;
  //   for (int j = i+1; j < natom; j++) {
  //     ex[i][j] = ex[j][i] = -(geom[i][0] - geom[j][0]) / get_bond(i, j);
  //     ey[i][j] = ey[j][i] = -(geom[i][1] - geom[j][1]) / get_bond(i, j);
  //     ez[i][j] = ez[j][i] = -(geom[i][2] - geom[j][2]) / get_bond(i, j);
  //   }
  // }

  // // calculate the dot product between two unit vectors spanning three atoms,
  // // then take the arccosine
  // for (int i = 0; i < natom; i++)
  //   for (int j = i+1; j < natom; i++)
  //     for (int k = j+1; k < natom; k++)
  // 	double eij = (ex[i][j]);
}

void Molecule::calc_oop_angles()
{

}

void Molecule::calc_torsion_angles()
{

}

void Molecule::calc_com(bool translatep)
{
  double mi;
  double M = 0;
  double CMx = 0;
  double CMy = 0;
  double CMz = 0;

  for (int i = 0; i < natom; i++) {
    mi = masses[(int)zvals[i]];
    M += mi;
    CMx += mi * geom[i][0];
    CMy += mi * geom[i][1];
    CMz += mi * geom[i][2];
  }

  CMx /= M; CMy /= M; CMz /= M;
  printf("\nMolecular center of mass [bohr]: %12.8f %12.8f %12.8f\n", CMx, CMy, CMz);

  if (translatep)
    translate(-CMx, -CMy, -CMz);
}

void Molecule::calc_moi()
{
  double mi, xi, yi, zi;
  // Calculate the moment of inertia tensor.
  for (int i = 0; i < natom; i++) {
    mi = masses[(int)zvals[i]];
    xi = geom[i][0]; yi = geom[i][1]; zi = geom[i][2];
    moi[0][0] += mi * ((yi*yi) + (zi*zi));
    moi[1][1] += mi * ((xi*xi) + (zi*zi));
    moi[2][2] += mi * ((xi*xi) + (yi*yi));
    moi[0][1] += mi * xi * yi;
    moi[0][2] += mi * xi * zi;
    moi[1][2] += mi * yi * zi;
  }
  moi[1][0] = moi[0][1];
  moi[2][0] = moi[0][2];
  moi[2][1] = moi[1][2];

  // Find the principal moments by diagonalizing the MOI tensor.
  // Unfortunately, we can't pass the diag routine a null pointer for
  // the eigenvectors, even though we don't need them.
  double **evecs = new double* [3];
  for (int i = 0; i < 3; i++) evecs[i] = new double[3];
  diag(3, 3, moi, moi_abc, false, evecs, 1e-13);
  for (int i = 0; i < 3; i++) delete[] evecs[i];
  delete[] evecs;

  // Print the moment of inertia tensor.
  printf("\nMoment of inertia tensor [amu][bohr]^2:\n");
  for (int i = 0; i < 3; i++)
    printf("%16.8f %16.8f %16.8f\n", moi[i][0], moi[i][1], moi[i][2]);

  // Print the principal moments of inertia.
  double conv;
  double amu2g = 1.6605402e-24;
  double bohr2ang = 0.529177249;
  printf("\nPrincipal moments of inertia:\n");
  printf("    [amu][bohr]^2: %12.8f %16.8f %16.8f\n", 
	 moi_abc[0], moi_abc[1], moi_abc[2]);
  conv = bohr2ang * bohr2ang;
  printf("[amu][angstrom]^2: %12.8f %16.8f %16.8f\n", 
	 moi_abc[0]*conv, moi_abc[1]*conv, moi_abc[2]*conv);
  conv = amu2g * bohr2ang * 1e-8 * bohr2ang * 1e-8;
  printf("        [g][cm]^2: %16.8e %16.8e %16.8e\n", 
	 moi_abc[0]*conv, moi_abc[1]*conv, moi_abc[2]*conv);

  print_rotor_type();
}

void Molecule::print_rotor_type()
{
  double A = moi_abc[0];
  double B = moi_abc[1];
  double C = moi_abc[2];

  if (natom == 2) printf("The molecule is diatomic.\n");
  else if (A < 1e-4)
    printf("The molecule is linear.\n");
  else if ((fabs(A-B) < 1e-4) && (fabs(B-C) < 1e-4))
    printf("The molecule is a spherical top.\n");
  else if ((fabs(A-B) < 1e-4) && (fabs(B-C) > 1e-4))
    printf("The molecule is an oblate symmetric top.\n");
  else if ((fabs(A-B) > 1e-4) && (fabs(B-C) < 1e-4))
    printf("The molecule is a prolate symmetric top.\n");
  else 
    printf("The molecule is an asymmetric top.\n");
}

void Molecule::calc_rot_const()
{
  double A = moi_abc[0];
  double B = moi_abc[1];
  double C = moi_abc[2];

  double conv;

  printf("\nRotational constants:\n");
  conv = rot_constant * (1/amu2kg) * (1/bohr2m) * (1/bohr2m) * speed_of_light * 1e-6;
  printf("  [MHz]: A: %16.8f B: %16.8f C: %16.8f\n", conv/A, conv/B, conv/C);
  conv = rot_constant * (1/amu2kg) * (1/bohr2m) * (1/bohr2m) / 100;
  printf("[cm]^-1: A: %16.8f B: %16.8f C: %16.8f\n", conv/A, conv/B, conv/C);
}
