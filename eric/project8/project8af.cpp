#include <arrayfire.h>

#include "../utils.hpp"

#include <iostream>

int main()
{

  // af::setBackend(AF_BACKEND_CUDA);
  af::info();

  const size_t NElec = 10;
  const size_t NOcc = NElec / 2;
  const size_t NBasis = 26;
  const size_t M = idx4(NBasis - 1, NBasis - 1, NBasis - 1, NBasis - 1) + 1;

  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  size_t l = 0;
  double val;
  size_t mu, nu, lam, sig;

  FILE *enuc_file;
  enuc_file = fopen("h2o_dzp_enuc.dat", "r");
  double Vnn;
  fscanf(enuc_file, "%lf", &Vnn);
  fclose(enuc_file);

  printf("Nuclear repulsion energy =  %12f\n", Vnn);

  af::array S(NBasis, NBasis, f64);
  af::array T(NBasis, NBasis, f64);
  af::array V(NBasis, NBasis, f64);

  // af::array H(NBasis, NBasis);
  af::array F = af::constant(0.0, NBasis, NBasis, f64);
  af::array F_prime = af::constant(0.0, NBasis, NBasis, f64);
  af::array D = af::constant(0.0, NBasis, NBasis, f64);
  af::array D_old = af::constant(0.0, NBasis, NBasis, f64);
  af::array C(NBasis, NBasis, f64);

  af::array eps_vec(NBasis, f64);
  af::array C_prime(NBasis, NBasis, f64);

  af::array Lam_S_vec(NBasis, f64);
  af::array Lam_S_mat(NBasis, NBasis, f64);
  af::array L_S(NBasis, NBasis, f64);

  FILE *S_file, *T_file, *V_file;
  S_file = fopen("h2o_dzp_s.dat", "r");
  T_file = fopen("h2o_dzp_t.dat", "r");
  V_file = fopen("h2o_dzp_v.dat", "r");

  while (fscanf(S_file, "%d %d %lf", &i, &j, &val) != EOF)
    S(i-1, j-1) = S(j-1, i-1) = val;
  while (fscanf(T_file, "%d %d %lf", &i, &j, &val) != EOF)
    T(i-1, j-1) = T(j-1, i-1) = val;
  while (fscanf(V_file, "%d %d %lf", &i, &j, &val) != EOF)
    V(i-1, j-1) = V(j-1, i-1) = val;

  fclose(S_file);
  fclose(T_file);
  fclose(V_file);

  af::array ERI = af::constant(0.0, M, f64);

  FILE *ERI_file;
  ERI_file = fopen("h2o_dzp_eri.dat", "r");

  while (fscanf(ERI_file, "%d %d %d %d %lf", &i, &j, &k, &l, &val) != EOF) {
    mu = i-1; nu = j-1; lam = k-1; sig = l-1;
    ERI(idx4(mu,nu,lam,sig)) = val;
  }

  fclose(ERI_file);

  const double thresh_E = 1.0e-15;
  const double thresh_D = 1.0e-10;
  size_t iteration = 0;
  const size_t max_iterations = 1024;
  double E_total, E_elec_old, E_elec_new, delta_E, rmsd_D;

  af::print("Overlap Integrals:", S);
  af::print("Kinetic-Energy Integrals:", T);
  af::print("Nuclear Attraction Integrals:", V);

  af::array H = T + V;

  af::print("Core Hamiltonian:", H);

  const bool af_has_lapack = af::isLAPACKAvailable();
  std::cout << "af_has_lapack: " << af_has_lapack << std::endl;

  // af::eigen(Lam_S_vec, L_S, S);
  // Lam_S_
  // af::array values, vectors;
  // af::eigen(values, vectors, S);

  return 0;

}
