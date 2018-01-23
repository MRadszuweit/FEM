/* lib: PCGS  */

#ifndef PCGS_H
#define PCGS_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include "linear_algebra.h"

int pcg(double* X,double* B,int n,double tol,int max_iter);
double get_pcg_residuum();

void PCG_set_precon(int (*Precon)(int n,double* R,double* Z),int max_precon);
void PCG_set_mat_mult(void (*Mat_mult)(int n,double* x,double* y));
void* PCG_get_precon();
void* PCG_get_mat_mult();
int PCG_get_precon_iter();

// AMG preconditioner 
int PCG_AMG_precon(int n,double* R,double* Z);
void PCG_AMG_mat_mult(int n,double* x,double* y);
void PCG_set_AMG_system(sparse_matrix* A,
		sparse_matrix** L,
		sparse_matrix** U,
		sparse_matrix** Restriction,
		sparse_matrix** Prolongation,
		sparse_matrix** Coarse_A, 
		double** C_L,
		double** C_U,
		int** Coarsening,
		int* Coarse_sizes,
		int* Fine_sizes,
		int depth);
void PCG_set_AMG_solver(double (*Prec_solver)(
		sparse_matrix** Coarse_A,
		sparse_matrix** Prolongation,
		sparse_matrix** Restriction,
		sparse_matrix** L,
		sparse_matrix** U,
		double** Coarsest_L,
		double** Coarsest_U,
		double* Sol,
		double* Fine_F,
		int** Coarsening,
		int* Fine_sizes,
		int* Coarse_sizes,
		int* Iterations,
		int max_depth,
		int depth));
void PCG_reset();
#endif
