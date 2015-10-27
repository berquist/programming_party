#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <armadillo>

// from diag.cpp:
void diag(int nm, int n, double **array, double *e_vals, int matz, double **e_vecs, double toler);
void tred2(int n, double **a, double *d, double *e, int matz);
void tqli(int n, double *d, double **z, double *e, int matz, double toler);
double *init_array(int size);
double **init_matrix(int n, int m);
void free_matrix(double **array, int size);
void eigsort(double *d, double **v, int n);
// void print_mat(double **a, int m, int n, FILE *out);
void zero_matrix(double **a, int m, int n);
void zero_array(double *a, int m);
void mmult(double **A, int transa, double **B, int transb, double **C, int l, int m, int n);

// from utils.cpp:
void print_mat(double **mat, int dim_rows, int dim_cols);
void print_arma_mat(const arma::mat& mat);
void print_arma_vec(const arma::vec& vec, int ncols);
int compound_index_2(int i, int j);
int compound_index_4(int i, int j, int k, int l);
int idx2(int i, int j);
int idx4(int i, int j, int k, int l);

#endif /* UTILS_HPP */
