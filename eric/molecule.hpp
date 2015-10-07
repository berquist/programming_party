#ifndef MOLECULE_HPP
#define MOLECULE_HPP

#include <string>

using namespace std;

class Molecule
{
public:
  int natom;
  int charge;
  double *zvals;
  double **geom;
  // double **ex, **ey, **ez;
  // double ***phi;
  double **moi, *moi_abc;
  string point_group;

  void print_geom();
  void print_bonds();
  void print_angles();
  void print_oop_angles();
  void print_torsion_angles();
  void print_com();
  void print_moi();
  void print_rot_const();

  void rotate(double phi);
  void translate(double x, double y, double z);
  double bond(int i, int j);
  double calc_unit(int i, int j, int cart);
  double angle(int i, int j, int k);
  double angle_oop(int i, int j, int k, int l);
  double angle_torsion(int i, int j, int k, int l);

  Molecule(int n, int q);
  ~Molecule();

private:
  double calc_bond(int i, int j);
  double calc_angle(int i, int j, int k);
  double calc_angle_oop(int i, int j, int k, int l);
  double calc_angle_torsion(int i, int j, int k, int l);
  void calc_bonds();
  void calc_angles();
  void calc_oop_angles();
  void calc_torsion_angles();
  void calc_com(bool translatep);
  void calc_moi();
  void calc_rot_const();
  void print_rotor_type();
};

#endif /* MOLECULE_HPP */
