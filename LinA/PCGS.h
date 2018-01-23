/* lib: PCGS  */

#ifndef PCGS_H
#define PCGS_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <omp.h>
#include "linear_algebra.h"
#include "Multigrid.h"

int pcg(double* X,double* B,int n,double tol,int max_iter);
double get_pcg_residuum();

void PCG_set_precon(int (*Precon)(int n,double* R,double* Z));
void PCG_set_mat_mult(void (*Mat_mult)(int n,double* x,double* y));
void* PCG_get_precon();
void* PCG_get_mat_mult();
void PCG_set_default_precon();
void PCG_default(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void PCG_set_AMG_preconditioner(sparse_matrix* A,int num_fields,int depth);

#endif
