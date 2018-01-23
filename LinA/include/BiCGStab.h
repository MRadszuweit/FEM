#ifndef BICGSTAB_H
#define BICGSTAB_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "linear_algebra.h"
#include "PCGS.h"
#include "Multigrid.h"

#define BI_NO 1
#define BI_YES 1

void BI_set_precon(int (*Precon)(int n,double* Y,double* X));
void BI_set_mat_mult(void (*Mat_mult)(int n,double* Y,double* X));
void BI_set_PCG_preconditioner(sparse_matrix* A);
void BI_set_AMG_preconditioner(sparse_matrix* A,int num_fields,int depth);
amg_system_info* BI_get_AMG_setup_data();

int BiCGStab_solve(double* X,double* b,int n);
double BI_residuum(double* X,double* b,int n);
void BI_set_eps(double eps);
void BI_set_max_iter(int max_iter);
int BI_get_max_iter();
int BI_AMG_precon(int n,double* Y,double* X);
void BI_set_AMG_setup_data(amg_system_info* AMG_data);
void BI_set_stat(char* Out_AMG,char* Out_BiCGStab);
amg_statistics* BI_get_AMG_stat();
amg_statistics* BI_get_stat();
void BI_print_info(int mode);
int BI_info_set();

#endif
