/* lib: GMRES_Newton  */

#ifndef GMRES_NEWTON
#define GMRES_NEWTON

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "linear_algebra.h"

#ifndef __cplusplus
	#include "itlin.h"
#endif

#define LARGE_SYSTEM 1
#define SMALL_SYSTEM 0

#ifndef __cplusplus
	int Newton_solver_mult_Root(sparse_matrix* Pattern,double maxiter,double eps,int mode);
	int Newton_solver(sparse_matrix* Pattern,double* X,double maxiter,double eps,int mode,double r0);
#endif

void Newton_set_function(double (*F)(double* X,int i,int size));
void Newton_set_inhomo(double* B,int n);
void Newton_GMRES_options(int bases_imax,int maxiter,double tol);
void Newton_first_guess_options(int guess_num,int size,double* Min,double* Max);
void Newton_set_range(int size,double* Min,double* Max);
sparse_matrix* Newton_get_full_pattern(int size);
double** Newton_get_all_roots();
void Newton_first_guess(double* Res,int n);
int Newton_1D(double (*F)(double x),double** Roots,double min,double max,int max_iter,double eps);
int Interval_Separation_Newton_1D(double (*F)(double x),double** Roots,double min,double max,int max_iter,double eps,int int_num);
void set_RN_Hesse(sparse_matrix* A);
void set_RN_linear(double* B);

#endif
