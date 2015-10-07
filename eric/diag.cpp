/* translation into C of a translation into FORTRAN77 of the EISPACK
 * matrix diagonalization routines
 */

#include <cstdio>
#include <cstdlib>
#include <strings.h>
#include <cmath>
#include "utils.hpp"

#define DSIGN(a,b) ((b) >= 0.0) ? (fabs(a)) : (-fabs(a))

/**
 * @brief Diagonalize a symmetric matrix and return its eigenvalues and
 * eigenvectors.
 *
 * @param int nm: The row/column dimension of the matrix.
 * @param int n:  Ditto.
 * @param double **array: The matrix
 * @param double *e_vals: A vector, which will contain the eigenvalues.
 * @param int matz: Boolean for returning the eigenvectors.
 * @param double **e_vecs: A matrix whose columns are the eigenvectors.
 * @param double toler: A tolerance limit for the iterative procedure.  Rec: 1e-13
 *
 */
void diag(int nm, int n, double **array, double *e_vals, int matz,
	  double **e_vecs, double toler)
{
  int i, j, ii, ij, ierr;
  int ascend_order;
  double *fv1, **temp;
  double zero = 0.0;
  double one = 1.0;

  /* Modified by Ed - matz can have the values 0 through 3 */

  if ((matz > 3) || (matz < 0)) {
    matz = 0;
    ascend_order = 1;
  }
  else
    if (matz < 2)
      ascend_order = 1;	/* Eigenvalues in ascending order */
    else {
      matz -= 2;
      ascend_order = 0;	/* Eigenvalues in descending order */
    }

  fv1 = (double *) init_array(n);
  temp = (double **) init_matrix(n,n);

  if (n > nm) {
    ierr = 10*n;
    fprintf(stderr,"n = %d is greater than nm = %d in rsp\n",n,nm);
    exit(ierr);
  }

  for (i=0; i < n; i++) {
    for (j=0; j < n; j++) {
      e_vecs[i][j] = array[i][j];
    }
  }

  tred2(n,e_vecs,e_vals,fv1,matz);

  for (i=0; i < n; i++)
    for (j=0; j < n; j++)
      temp[i][j]=e_vecs[j][i];

  tqli(n,e_vals,temp,fv1,matz,toler);

  for (i=0; i < n; i++)
    for (j=0; j < n; j++)
      e_vecs[i][j]=temp[j][i];

  if (ascend_order)
    eigsort(e_vals,e_vecs,n);
  else
    eigsort(e_vals,e_vecs,(-1)*n);

  free(fv1);
  free_matrix(temp,n);

}

/**
 * @brief converts symmetric matrix to a tridiagonal form for use in tqli
 *
 * if matz = 0, only find eigenvalues
 * else find both eigenvalues and eigenvectors
 *
 * @param[in] n
 * @param[] a
 * @param[] d
 * @param[] e
 * @param[in] matz
 */
void tred2(int n, double **a, double *d, double *e, int matz)
{
  int i,j,k,l,il,ik,jk,kj;
  double f,g,h,hh,scale,scale_inv,h_inv;
  double temp;

  if (n == 1) return;

  for (i=n-1; i > 0; i--) {
    l = i-1;
    h = 0.0;
    scale = 0.0;
    if (l) {
      for (k=0; k <= l; k++) {
	scale += fabs(a[i][k]);
      }
      if (scale == 0.0) {
	e[i] = a[i][l];
      }
      else {
	scale_inv=1.0/scale;
	for (k=0; k <= l; k++) {
	  a[i][k] *= scale_inv;
	  h += a[i][k]*a[i][k];
	}
	f=a[i][l];
	g= -(DSIGN(sqrt(h),f));
	e[i] = scale*g;
	h -= f*g;
	a[i][l] = f-g;
	f = 0.0;
	h_inv=1.0/h;
	for (j=0; j <= l; j++) {
	  if (matz) a[j][i] = a[i][j]*h_inv;
	  g = 0.0;
	  for (k=0; k <= j; k++) {
	    g += a[j][k]*a[i][k];
	  }
	  if (l > j) {
	    for (k=j+1; k <= l; k++) {
	      g += a[k][j]*a[i][k];
	    }
	  }
	  e[j] = g*h_inv;
	  f += e[j]*a[i][j];
	}
	hh = f/(h+h);
	for (j=0; j <= l; j++) {
	  f = a[i][j];
	  g = e[j] - hh*f;
	  e[j] = g;
	  for (k=0; k <= j; k++) {
	    a[j][k] -= (f*e[k] + g*a[i][k]);
	  }
	}
      }
    }
    else {
      e[i] = a[i][l];
    }
    d[i] = h;
  }
  if (matz) d[0] = 0.0;
  e[0] = 0.0;

  for (i=0; i < n; i++) {
    l = i-1;
    if (matz) {
      if (d[i]) {
	for (j=0; j <= l; j++) {
	  g = 0.0;
	  for (k=0; k <= l; k++) {
	    g += a[i][k]*a[k][j];
	  }
	  for (k=0; k <= l; k++) {
	    a[k][j] -= g*a[k][i];
	  }
	}
      }
    }
    d[i] = a[i][i];
    if (matz) {
      a[i][i] = 1.0;
      if (l >= 0) {
	for (j=0; j<= l; j++) {
	  a[i][j] = 0.0;
	  a[j][i] = 0.0;
	}
      }
    }
  }
}

/**
 * @brief diagonalizes tridiagonal matrix output by tred2
 *
 * gives only eigenvalues if matz = 0
 * gives both eigenvalues and eigenvectos if matz = 1
 *
 * @param[in] n
 * @param[in,out] d
 * @param[in,out] z
 * @param[in,out] e
 * @param[in] matz
 * @param[in] toler
 */
void tqli(int n, double *d, double **z, double *e, int matz, double toler)
{
  register int k;
  int i,j,l,m,iter;
  double dd,g,r,s,c,p,f,b,h;
  double azi;

  f=0.0;
  if (n == 1) {
    d[0]=z[0][0];
    z[0][0] = 1.0;
    return;
  }

  for (i=1; i < n ; i++) {
    e[i-1] = e[i];
  }
  e[n-1] = 0.0;
  for (l=0; l < n; l++) {
    iter = 0;
  L1:
    for (m=l; m < n-1;m++) {
      dd = fabs(d[m]) + fabs(d[m+1]);
#if 0
      if (fabs(e[m])+dd == dd) goto L2;
#else
      if (fabs(e[m]) < toler) goto L2;
#endif
    }
    m=n-1;
  L2:
    if (m != l) {
      if (iter++ == 30) {
	fprintf (stderr,"tqli not converging\n");
	continue;
#if 0
	exit(30);
#endif
      }

      g = (d[l+1]-d[l])/(2.0*e[l]);
      r = sqrt(g*g + 1.0);
      g = d[m] - d[l] + e[l]/((g + (DSIGN(r,g))));
      s=1.0;
      c=1.0;
      p=0.0;
      for (i=m-1; i >= l; i--) {
	f = s*e[i];
	b = c*e[i];
	if (fabs(f) >= fabs(g)) {
	  c = g/f;
	  r = sqrt(c*c + 1.0);
	  e[i+1] = f*r;
	  s=1.0/r;
	  c *= s;
	}
	else {
	  s = f/g;
	  r = sqrt(s*s + 1.0);
	  e[i+1] = g*r;
	  c = 1.0/r;
	  s *= c;
	}
	g = d[i+1] - p;
	r = (d[i]-g)*s + 2.0*c*b;
	p = s*r;
	d[i+1] = g+p;
	g = c*r-b;

	if (matz) {
	  double *zi = z[i];
	  double *zi1 = z[i+1];
	  for (k=n; k ; k--,zi++,zi1++) {
	    azi = *zi;
	    f = *zi1;
	    *zi1 = azi*s + c*f;
	    *zi = azi*c - s*f;
	  }
	}
      }

      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
      goto L1;
    }
  }
}


/**
 * @brief allocate memory for a len(size) array, return pointer to first element
 *
 * @param[in] size len(array)
 */
double *init_array(int size)
{
  double *array;

  if ((array = (double *) malloc(sizeof(double)*size)) == NULL) {
    fprintf(stderr, "init_array:  trouble allocating memory \n");
    fprintf(stderr, "size = %d\n", size);
    exit(2);
  }
  bzero(array, sizeof(double)*size);
  return(array);
}

/**
 * @brief allocate memory for an n*m matrix, return pointer to pointer to first element
 *
 * @param[in] n # rows in matrix
 * @param[in] m # cols in matrix
 */
double **init_matrix(int n, int m)
{
  double **array = NULL;
  int i;

  if ((array = (double **) malloc(sizeof(double *)*n)) == NULL) {
    fprintf(stderr, "init_matrix: trouble allocating memory \n");
    fprintf(stderr, "n = %d\n", n);
    exit(2);
  }

  for (i = 0; i < n; i++) {
    if ((array[i] = (double *) malloc(sizeof(double)*m)) == NULL) {
      fprintf(stderr, "init_matrix: trouble allocating memory \n");
      fprintf(stderr, "i = %d m = %d\n", i, m);
      exit(3);
    }
    bzero(array[i], sizeof(double)*m);
  }
  return(array);
}

/**
 * @brief Free the memory used by an array, deleting it.
 *
 * @param[in,out] array array to be deleted
 * @param[in] size len(array)
 */
void free_matrix(double **array, int size)
{
  int i;

  for (i = 0; i < size; i++) {
    free(array[i]);
  }

  free(array);
}

/**
 * @brief
 *
 * @param[]
 * @param[]
 * @param[]
 */
void eigsort(double *d, double **v, int n)
{
  int i,j,k;
  double p;

  /* Modified by Ed - if n is negative - sort eigenvalues in descending order */

  if (n >= 0) {
    for (i = 0; i < n-1; i++) {
      k = i;
      p = d[i];
      for (j = i+1; j < n; j++) {
	if (d[j] < p) {
	  k = j;
	  p = d[j];
	}
      }
      if (k != i) {
	d[k] = d[i];
	d[i] = p;
	for (j = 0; j < n; j++) {
	  p = v[j][i];
	  v[j][i] = v[j][k];
	  v[j][k] = p;
	}
      }
    }
  }
  else {
    n = abs(n);
    for (i = 0; i < n-1; i++) {
      k = i;
      p = d[i];
      for (j = i+1; j < n; j++) {
	if (d[j] > p) {
	  k = j;
	  p = d[j];
	}
      }
      if (k != i) {
	d[k] = d[i];
	d[i] = p;
	for (j = 0; j < n; j++) {
	  p = v[j][i];
	  v[j][i] = v[j][k];
	  v[j][k] = p;
	}
      }
    }
  }
}

/**
 * @brief
 *
 * @param[in] a
 * @param[in] m
 * @param[in] n
 * @param[] out
 */
void print_mat(double **a, int m, int n, FILE *out)
{
  int ii,jj,kk,nn,ll;
  int i,j,k;

  ii=0;jj=0;
 L200:
  ii++;
  jj++;
  kk=10*jj;
  nn=n;
  if (nn > kk) nn=kk;
  ll = 2*(nn-ii+1)+1;
  fprintf (out,"\n");
  for (i=ii; i <= nn; i++) fprintf(out,"       %5d",i);
  fprintf (out,"\n");
  for (i=0; i < m; i++) {
    fprintf (out,"\n%5d",i+1);
    for (j=ii-1; j < nn; j++) {
      fprintf (out,"%12.7f",a[i][j]);
    }
  }
  fprintf (out,"\n");
  if (n <= kk) {
    fflush(out);
    return;
  }
  ii=kk; goto L200;
}

/**
 * @brief Zero out a matrix.
 *
 * @param[in,out] a matrix to be zeroed
 * @param[in] m # of rows in a
 * @param[in] n # of cols in a
 */
void zero_matrix(double **a, int m, int n)
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      a[i][j] = 0.0;
}

/**
 * @brief Zero out an array (vector).
 *
 * @param[in,out] a array to be zeroed
 * @param[in] m len(a)
 */
void zero_array(double *a, int m)
{
  int i;

  for (i = 0; i < m; i++)
    a[i] = 0.0;
}

/**
 * @brief Perform the matrix operation \f$C = (A*B) + C\f$.
 *
 * transa = 0, transb = 0: C += A * B
 * transa = 1, transb = 0: C += A.t * B
 * transa = 0, transb = 1: C += A * B.t
 * transa = 1, transb = 1: C += A.t * B.t
 *
 * @param[in] A left matrix** to be multiplied
 * @param[in] transa 0 for A, 1 for A.t
 * @param[in] B right matrix** to be multiplied
 * @param[in] transb 0 for B, 1 for B.t
 * @param[in,out] C resulting matrix**
 * @param[in] l # of rows of C (rows A)
 * @param[in] m len of contraction dimension (cols A, rows B)
 * @param[in] n # of cols of C (cols B)
 */
void mmult(double **A, int transa, double **B, int transb, double **C, int l, int m, int n)
{
  int i, j, k;
 
  if (!transa && !transb) {
    for (i = 0; i < l; i++) {
      for (j = 0; j < m; j++) {
        for (k = 0; k < n; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  else if (!transa && transb) {
    for (i = 0; i < l; i++) {
      for (j = 0; j < m; j++) {
        for (k = 0; k < n; k++) {
          C[i][j] += A[i][k] * B[j][k];
        }
      }
    }
  }
  else if (transa && !transb) {
    for (i = 0; i < l; i++) {
      for (j = 0; j < m; j++) {
        for (k = 0; k < n; k++) {
          C[i][j] += A[k][i] * B[k][j];
        }
      }
    }
  }
  else if (transa && transb) {
    for (i = 0; i < l; i++) {
      for (j = 0; j < m; j++) {
        for (k = 0; k < n; k++) {
          C[i][j] += A[k][i] * B[j][k];
        }
      }
    }
  }

}
