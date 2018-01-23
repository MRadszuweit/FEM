#ifndef REF_NEWTON
#define REF_NEWTON

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "linear_algebra.h"
#include "PCGS.h"
#include "Multigrid.h"
#include "GMRES_Newton.h"
#include "BiCGStab.h"

#define POSITIVE_DEFINITE 1
#define INDEFINITE 0
#define RN_NO 0
#define RN_YES 1
#define RN_FAILED 0

#define RN_BICGSTAB 1
#define RN_AMG 2
#define RN_PCG 3
#define RN_CHOLESKY 4
#define RN_UMF 5


///////////////////////// Datentypen ////////////////////////////////////////////

typedef struct RN_STATISTICS{
	int calls;
	int average_eig_iter;
	int number_neg_curv;
	int line_search_count;
	int line_search_iter;
	int total_iter;
	double res;
}RN_statistics;

///////////////////////// Funktionen /////////////////////////////////////

void RN_set_params(
	int R_Newton_max_iter,
	double R_Newton_eps,
	double R_Newton_sigma_l,
	double R_Newton_sigma_u,
	double R_Newton_rho,
	double R_Newton_Delta,
	double R_Newton_Xi,
	double R_Newton_trust_lower,
	double R_Newton_trust_upper,
	double R_Newton_trust_radius_min,
	double R_Newton_trust_radius_max);

int simulated_annealing_min(double* X,int n);
void set_RN_params(int max_iter,double eps, double sigma_l, double sigma_u);
void set_RN_constraints(int* Index_set, double* Lower, double* Upper, int n);
void set_RN_Hesse(sparse_matrix* A); 
void set_RN_linear(double* B);
void set_RN_functional(double (*F)(double* X,int n));
void RN_set_kernel(double* K,int n);
void subspace_minimum(sparse_matrix* A,double* b,double* V1,double* V2,double* X);
int reflective_newton(double* X);
void RN_print_function(double (*F)(double x),int n,double min,double max);
void print_RN_statistics(FILE* file);
//void set_aux_constraints(int* Indices, int size);
//void include_aux_constraints(sparse_matrix* A,double* b,int* Indices,int size);
void RN_set_default_matrix(sparse_matrix* M);
void RN_default_mat_mult(int n,double* y,double* z);
void RN_set_gradient(double* (*Gradient_func)(double* X));
void RN_set_Hesse_function(sparse_matrix* (*Hesse_func)(double* X),int dim_Hesse);
void RN_set_AMG_data(amg_system_info* AMG_info);
int* get_all_indices(int size);
void RN_print_info(int mode);
int RN_info_set();
void RN_set_solver(int type);
void RN_set_local_solver(int (*Solver)(sparse_matrix* A,double* X,double* b));
void RN_set_trust_region(double R_Newton_Delta);
void RN_set_trust_limits(double min,double max);
void RN_set_debug_mode(int mode);

#endif
