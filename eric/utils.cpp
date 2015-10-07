#include <cstdio>
#include "utils.hpp"

/// Pretty-print a general matrix in a traditional format.
void print_mat(double **mat, int dim_rows, int dim_cols) {
  // first, handle the column labels
  printf("     ");
  for (int i = 0; i < dim_cols; i++)
    printf("%12d", i+1);
  printf("\n");
  // then, handle the row labels
  for (int i = 0; i < dim_rows; i++) {
    printf("%5d", i+1);
    // print the matrix data
    for (int j = 0; j < dim_cols; j++)
      printf("%12.7f", mat[i][j]);
    printf("\n");
  }
}

void print_arma_mat(const arma::mat& mat) {
  // first, handle the column labels
  printf("     ");
  for (int i = 0; i < mat.n_cols; i++)
    printf("%12d", i+1);
  printf("\n");
  // then, handle the row labels
  for (int i = 0; i < mat.n_rows; i++) {
    printf("%5d", i+1);
    // print the matrix data
    for (int j = 0; j < mat.n_cols; j++)
      printf("%12.7f", mat(i,j));
    printf("\n");
  }
}

void print_arma_vec(const arma::vec& vec, int ncols) {
  // print ncols elements of vec before moving to the next row
  int counter = 0;
  while (counter < vec.n_elem) {
    printf("     ");
    for (int i = counter; i < counter + ncols && i < vec.n_elem; i++)
      printf("%12.7f", vec(i));
    printf("\n");
    counter += ncols;
  }
}

int compound_index_2(int i, int j) {
  if (i < j)
    return i*(i+1)/2 + j;
  else
    return j*(j+1)/2 + i;
}

int compound_index_4(int i, int j, int k, int l) {
  int ij = compound_index_2(i,j);
  int kl = compound_index_2(k,l);
  return compound_index_2(ij,kl);
}

arma::Col<int> gen_ioff(int M) {
  arma::Col<int> ioff = arma::Col<int>(M);
  ioff(0) = 0;
  for (int i = 1; i < ioff.n_elem; i++)
    ioff(i) = ioff(i-1) + i;
  return ioff;
}

int idx2(int i, int j) {
  if (i > j)
    // return ioff(i) + j;
    return i*(i+1)/2 + j;
  else
    // return ioff(j) + i;
    return j*(j+1)/2 + i;
}

int idx4(int i, int j, int k, int l) {
  int ij = idx2(i,j);
  int kl = idx2(k,l);
  return idx2(ij,kl);
}
