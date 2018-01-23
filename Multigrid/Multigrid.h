/* lib Multigrid*/

#ifndef MULTI_H
#define MULTI_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <time.h>
#include <mcheck.h>
#include <omp.h> 
#include "linear_algebra.h"
#include "red_black_tree.h"
#include "PCGS.h"

#define NO 0
#define YES 1
#define UP 1
#define DOWN -1
#define NONE 0
#define LEFT 1
#define RIGHT 2
#define UNDECIDED 0
#define COARSE 1
#define FINE 2
#define ISOLATED 3
#define WEAK 4 
#define DIRECT 1
#define STANDARD 2
#define AGGRESSIVE 1
#define WITHOUT_IOSOLATED 0
#define WITH_ISOLATED 1
#define STAND_ALONE 0
#define PREMODE 1
#define AMG_PCG 1
#define FULL_AMG 2


// types

typedef struct AMG_SYSTEM_INFO{
	sparse_matrix** Coarse_A;
	sparse_matrix** Prolongation;
	sparse_matrix** Restriction;
	sparse_matrix** L;
	sparse_matrix** U;
	sparse_matrix* Signs;
	int** Coarsening;
	int* Fine_sizes;
	int* Coarse_sizes;
	int depth;
	int AMG_cycle_structure;
	int AMG_prolongation_method;
	int AMG_smoothing_iterations;
	int AMG_aggr_smoothing_iterations;
	int AMG_coarsening;
	int AMG_info;
	int AMG_deg_of_freedom;
	int (*F_Prolong)(sparse_matrix* A,sparse_matrix* B,int* Label,int i,
 	 int** Indices,double** Values,int** Status,int** Strong,int** Positions);
	void (*AMG_coarsest_solver)(sparse_matrix* L,sparse_matrix* U,double* F,double* Sol,int size);
	int* (*Get_iter)(int depth);
 }amg_system_info;

// functions

double AMG_solve(amg_system_info* System_data,double* F,double* Fine_Sol);

amg_system_info* AMG_setup(
	sparse_matrix* A,
	int AMG_deg_of_freedom,
	int depth,
	int AMG_smoothing_iterations,
	int AMG_aggr_smoothing_iterations,
	int AMG_coarsening,
	int AMG_cycle_structure,
	int AMG_prolongation_method,
	void (*AMG_coarsest_solver)(sparse_matrix* L,sparse_matrix* U,double* F,double* Sol,int size),
	int* (*Get_iter)(int depth));
int* AMG_default_iterations(int depth);
void AMG_print_info(int s);
void AMG_free_data(amg_system_info* Data);
void AMG_set_SOR_relaxation_coeff(double alpha);
void set_connection_thresholds(double negative,double positive);

#endif
