#include <deque>

#include "../utils.hpp"

/*!
 * @brief Calculate the Hartree-Fock electronic energy.
 */
double calc_elec_energy(arma::mat& P, arma::mat& H, arma::mat& F) {
  return arma::accu(P % (H + F));
}

/*!
 * @brief Form the density matrix from the MO coefficients.
 */
void build_density(arma::mat& P, arma::mat& C, size_t NOcc) {
  P = C.cols(0, NOcc-1) * C.cols(0, NOcc-1).t();
}

/*!
 * @brief Build the Fock matrix from the density, one-electron,
 * and two-electron integrals.
 */
void build_fock(arma::mat& F, arma::mat& P, arma::mat& H, arma::vec& ERI) {
  for (size_t mu = 0; mu < H.n_rows; mu++) {
    for (size_t nu = 0; nu < H.n_cols; nu++) {
      F(mu, nu) = H(mu, nu);
      for (size_t lm = 0; lm < P.n_rows; lm++) {
        for (size_t sg = 0; sg < P.n_cols; sg++) {
          F(mu, nu) += P(lm, sg) * (2*ERI(idx4(mu, nu, lm, sg)) - ERI(idx4(mu, lm, nu, sg)));
        }
      }
    }
  }
}

/*!
 * @brief Calculate the RMS deviation between two density matrices.
 */
double rmsd_density(arma::mat& P_new, arma::mat& P_old) {
  return sqrt(arma::accu(arma::pow((P_new - P_old), 2)));
}

/*!
 * @brief Perform Hartree "damping" by mixing a fraction of old density
 * with the new density to aid convergence.
 */
void mix_density(arma::mat& P_new, arma::mat& P_old, double alpha) {
  // alpha must be in the range [0, 1)
  P_new = ((1.0 - alpha) * P_new) + (alpha * P_old);
}

/*!
 * @brief Build the DIIS error matrix.
 *
 * The formula for the error matrix at the \textit{i}th iteration is:
 *  $e_{i}=F_{i}D_{i}S-SD_{i}F_{i}$
 */
arma::mat build_error_matrix(arma::mat& F,
                             arma::mat& D,
                             arma::mat& S) {
  return (F*D*S) - (S*D*F);
}

/*!
 * @brief Build the DIIS B matrix, or "$A$" in $Ax=b$.
 */
arma::mat build_B_matrix(std::deque< arma::mat >& e) {
  size_t NErr = e.size();
  arma::mat B(NErr + 1, NErr + 1);
  B(NErr, NErr) = 0.0;
  for (size_t a = 0; a < NErr; a++) {
    B(a, NErr) = B(NErr, a) = -1.0;
    for (size_t b = 0; b < a + 1; b++)
      B(a, b) = B(b, a) = arma::dot(e[a].t(), e[b]);
  }
  return B;
}

/*!
 * @brief Build the extrapolated Fock matrix from the Fock vector.
 *
 * The formula for the extrapolated Fock matrix is:
 *  $F^{\prime}=\sum_{k}^{m}c_{k}F_{k}$
 * where there are $m$ elements in the Fock and error vectors.
 */
void build_extrap_fock(arma::mat& F_extrap,
                       arma::vec& diis_coeffs,
                       std::deque< arma::mat >& diis_fock_vec) {
  size_t len = diis_coeffs.n_elem - 1;
  F_extrap.zeros();
  for (size_t i = 0; i < len; i++)
    F_extrap += (diis_coeffs(i) * diis_fock_vec[i]);
  return;
}

/*!
 * @brief Build the DIIS "zero" vector, or "$b$" in $Ax=b$.
 */
arma::vec build_diis_zero_vec(size_t len) {
  arma::vec diis_zero_vec(len, arma::fill::zeros);
  diis_zero_vec(len - 1) = -1.0;
  return diis_zero_vec;
}

int main()
{

  size_t NElec = 10;
  size_t NOcc = NElec / 2;
  size_t NBasis = 26;
  size_t M = idx4(NBasis - 1, NBasis - 1, NBasis - 1, NBasis - 1) + 1;

  size_t i, j, k, l;
  double val;
  size_t mu, nu, lam, sig;

  FILE *enuc_file;
  enuc_file = fopen("h2o_dzp_enuc.dat", "r");
  double Vnn;
  fscanf(enuc_file, "%lf", &Vnn);
  fclose(enuc_file);

  printf("Nuclear repulsion energy =  %12f\n", Vnn);

  arma::mat S(NBasis, NBasis);
  arma::mat T(NBasis, NBasis);
  arma::mat V(NBasis, NBasis);
  arma::mat H(NBasis, NBasis);
  arma::mat F(NBasis, NBasis, arma::fill::zeros);
  arma::mat F_prime(NBasis, NBasis, arma::fill::zeros);
  arma::mat D(NBasis, NBasis, arma::fill::zeros);
  arma::mat D_old(NBasis, NBasis, arma::fill::zeros);
  arma::mat C(NBasis, NBasis);

  arma::vec eps_vec(NBasis);
  arma::mat C_prime(NBasis, NBasis);

  arma::vec Lam_S_vec(NBasis);
  arma::mat Lam_S_mat(NBasis, NBasis);
  arma::mat L_S(NBasis, NBasis);

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

  arma::vec ERI = arma::vec(M, arma::fill::zeros);

  FILE *ERI_file;
  ERI_file = fopen("h2o_dzp_eri.dat", "r");

  while (fscanf(ERI_file, "%d %d %d %d %lf", &i, &j, &k, &l, &val) != EOF) {
    mu = i-1; nu = j-1; lam = k-1; sig = l-1;
    ERI(idx4(mu,nu,lam,sig)) = val;
  }

  fclose(ERI_file);

  double thresh_E = 1.0e-15;
  double thresh_D = 1.0e-10;
  size_t iteration = 0;
  size_t max_iterations = 1024;
  double E_total, E_elec_old, E_elec_new, delta_E, rmsd_D;

  printf("Overlap Integrals:\n");
  print_arma_mat(S);
  printf("Kinetic-Energy Integrals:\n");
  print_arma_mat(T);
  printf("Nuclear Attraction Integrals\n");
  print_arma_mat(V);

  H = T + V;

  printf("Core Hamiltonian:\n");
  print_arma_mat(H);

  arma::eig_sym(Lam_S_vec, L_S, S);
  // What's wrong with this?
  // Lam_S_mat = Lam_S_vec * arma::eye<arma::mat>(Lam_S_vec.n_elem, Lam_S_vec.n_elem);
  Lam_S_mat = arma::diagmat(Lam_S_vec);
  arma::mat Lam_sqrt_inv = arma::sqrt(arma::inv(Lam_S_mat));
  arma::mat symm_orthog = L_S * Lam_sqrt_inv * L_S.t();
  F_prime = symm_orthog.t() * H * symm_orthog;
  arma::eig_sym(eps_vec, C_prime, F_prime);
  C = symm_orthog * C_prime;
  build_density(D, C, NOcc);

  printf("S^-1/2 Matrix:\n");
  print_arma_mat(symm_orthog);
  printf("Initial F' Matrix:\n");
  print_arma_mat(F_prime);
  printf("Initial C Matrix:\n");
  print_arma_mat(C);
  printf("Initial Density Matrix:\n");
  print_arma_mat(D);

  E_elec_new = calc_elec_energy(D, H, H);
  E_total = E_elec_new + Vnn;
  delta_E = E_total;
  printf(" Iter        E(elec)              E(tot)               Delta(E)             RMS(D)\n");
  printf("%4d %20.12f %20.12f %20.12f\n",
         iteration, E_elec_new, E_total, delta_E);
  iteration++;

  /*!
   * Prepare structures necessary for DIIS extrapolation.
   */
  size_t NErr;
  std::deque< arma::mat > diis_error_vec;
  std::deque< arma::mat > diis_fock_vec;
  size_t max_diis_length = 6;
  arma::mat diis_error_mat;
  arma::vec diis_zero_vec;
  arma::mat B;
  arma::vec diis_coeff_vec;

  /*!
   * Start the SCF iterative procedure
   */
  while (iteration < max_iterations) {
    build_fock(F, D, H, ERI);
    /// Start collecting elements for DIIS once we're past the first iteration.
    if (iteration > 0) {
      diis_error_mat = build_error_matrix(F, D, S);
      NErr = diis_error_vec.size();
      if (NErr >= max_diis_length) {
        diis_error_vec.pop_back();
        diis_fock_vec.pop_back();
      }
      diis_error_vec.push_front(diis_error_mat);
      diis_fock_vec.push_front(F);
      NErr = diis_error_vec.size();
      /// Perform DIIS extrapolation only if we have 2 or more points.
      if (NErr >= 2) {
        diis_zero_vec = build_diis_zero_vec(NErr + 1);
        B = build_B_matrix(diis_error_vec);
        diis_coeff_vec = arma::solve(B, diis_zero_vec);
        build_extrap_fock(F, diis_coeff_vec, diis_fock_vec);
      }
    }
    F_prime = symm_orthog.t() * F * symm_orthog;
    arma::eig_sym(eps_vec, C_prime, F_prime);
    C = symm_orthog * C_prime;
    D_old = D;
    build_density(D, C, NOcc);
    E_elec_old = E_elec_new;
    E_elec_new = calc_elec_energy(D, H, F);
    E_total = E_elec_new + Vnn;
    if (iteration == 1) {
      printf("Fock Matrix:\n");
      print_arma_mat(F);
      printf("%4d %20.12f %20.12f %20.12f\n",
             iteration, E_elec_new, E_total, delta_E);
    } else {
      printf("%4d %20.12f %20.12f %20.12f %20.12f\n",
             iteration, E_elec_new, E_total, delta_E, rmsd_D);
    }
    delta_E = E_elec_new - E_elec_old;
    rmsd_D = rmsd_density(D, D_old);
    if (delta_E < thresh_E && rmsd_D < thresh_D) {
      printf("Convergence achieved.\n");
      break;
    }
    F = F_prime;
    iteration++;
  }

  arma::mat F_MO = C.t() * F * C;

  // Save the TEIs and MO coefficients/energies to disk
  // for use in other routines.
  H.save("H.mat", arma::arma_ascii);
  ERI.save("TEI_AO.mat", arma::arma_ascii);
  C.save("C.mat", arma::arma_ascii);
  F_MO.save("F_MO.mat", arma::arma_ascii);

  return 0;

}
