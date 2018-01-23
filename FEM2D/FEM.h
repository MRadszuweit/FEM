/* lib: FEM  */

#ifndef FEM_H
#define FEM_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include <mcheck.h>
#include <stdarg.h>
#include "FEM2D.h"
#include "linear_algebra.h"
#include "File_stuff.h"
#include "itlin.h"
#include "PCGS.h"
#include "Multigrid.h"

#define NONE 0
#define DIRICHLET 1
#define NEUMANN 2
#define ROBIN 3

#define DIAGONAL 1
#define ILU 2

#define GMRES 1
#define GAUSS_SEIDEL 2
#define PCG 3
#define ILU_PCG 4

#define MATRIX 1
#define SOLVER 2

#define ZERO -1

// Typen /////////////////////////////////////////////////

typedef void (*OP_type)(double*,double*,double,double);
typedef int (*PRECON_type)(int,double*,double*);
typedef void (*MULT_type)(int,double*,double*);

//  Funktionen ///////////////////////////////////////////

void run_FEM_simulation(char* name,char* output_name,char* mesh_name,char* mesh_dir,int steps,
 int interval,double time_length,char* initial_file_name);

void set_partial_solvers(void** Solvers);

void main_solve(int steps,double time_length,FILE* file,int interval,char* init_file);

void set_boundary_equations(
	int (*cond)(int j_vertex,int k_var),
	double (*f_dirichlet)(int i,int n,double t,double dt),
	double (*f_neumann)(int i,int n,double t,double dt),
	double (*f_robin)(int i,int n,double t,double dt));

void set_init_procedure(void (*F_init_System)(void* Params));

void set_var_blocks(int** (*get_var_blocks)(int** Len,int blocks),int blocks);

void set_initial_equations(double (*f_init)(int i_vertex,int var_index));

void boundary_conditions(sparse_matrix* A,double* F,int** conditions,double t,double dt);

int** FEM_get_boundary_conditions(int mesh_index,int block_index);

void set_dirichlet(sparse_matrix* A,double* F,int** conditions,double t,double dt,int zero_mode);
void set_neumann(sparse_matrix* A,double* F,int** conditions,int varnum,double t,double dt,int zero_mode);
void set_robin(sparse_matrix* A,double* F,int** conditions,int varnum,double t,double dt);
void set_direct_dirichlet(sparse_matrix* A,double* F,int** conditions,double* Values);
void delete_dirichlet_components(double* F,int** conditions,int size);


void partial_decoupling_solver(double* Solution,double* Old_solution,double t,double dt);
#ifndef __cplusplus

	int FEM_solver(double* F,double* Sol,...);
	void power_inversion(double* X,sparse_matrix* A,int power);
	void adaptive_inversion(double* X,sparse_matrix* A,double initial_dt);
#endif


//void multi_solver(double* Solution,double* F,int mesh_number,int varnum,
//sparse_matrix* (*set_matrix)(double* F,int mesh_index,int mesh_size));

//int adaptive_solver(double* Solution,double* Old_solution,int mesh_number,double t,double dt,
//	void (*Method)(sparse_matrix* matrix,double* vector,double* Prev,double t,double dt));

//int sub_solver(double* Solution,double* Old_solution,int mesh_number,double t,double dt,
//void (*Method)(sparse_matrix* matrix,double* vector,double* Prev,double* Prev_old,double t,double dt));

int preconlr(int n,double *z,double* w);
int gauss_seidel_precon(int n,double* z,double* w);
int solver(double* F,double* Sol,int (*Precon)(int n,double *z,double* w));
void Runge_Kutta4(double* X,int size,int var_num,double t,double dt,int steps,
 double (*F)(double* Y,int mesh_index,double t,int var_index));
double get_adaptive_time_step(double dt,sparse_matrix* A,double* X);

void GMRES_iteration_control(int iter);
void set_preconditioner(sparse_matrix* A,int mode);
void set_solver_mode(int mode);
void matvec(int n,double* y,double* z);
void set_var_number(int var_num);
void set_system_matrix(sparse_matrix* A);
FILE* init_file();
void init_solver(int i_max,int maxiter,int GS_maxiter,double tol);
void free_globals();
void set_solver_prec(double prec);
double get_solver_prec();
int get_bound_num();
double* divergence(double* Solution,int n_sol,int var);
double** gradient(double* Solution,int n_sol,int var);
sparse_matrix* set_Laplace(double* F,int mesh_index,int mesh_size);
double* get_function(double* Sol,int mesh_size,int varnum,int equ,double (*Func)(double* X,int k));
double* get_matrix_function(double* Sol,int mesh_size,int varnum,int equ,
 double (*Func)(double* X,int k_mesh,int k_index));
sparse_matrix* FEM_get_dirichlet_correction(sparse_matrix* A,int** conditions, double diag);
void FEM_impose_dirichlet(double* Solution,int size,int** conditions,
 double (*F_dirichlet)(int i_mesh,int n_mesh,double t,double dt),double t,double dt);
void FEM_add_noise(double* Sol,double amplitude,int m,int var_index);

void test_mem();
void init_log_file();
void disable_log_file();
int cmp_sol(int mesh_number,double*** Solutions);
void insert_index(int** Connections,int n,int* Len,int ind);
void read_node_file2D(char* path,char* Name,mesh2D* Mesh,double** Attr,int* attr_num,int quiet);
int read_poly_file(char* path,char* Name,int** Borders,int* Size0,int* Size1,point2D** Inside,int quiet);
void read_element_file2D(char* path,char* Name,mesh2D* Mesh,index3D** Elements,int* tri_num,int quiet);
void set_boundary_marker(mesh2D* Mesh);
void set_boundary_equation(sparse_matrix* A,double* b,int equ,int var,double factor);
double get_var_number();
//double* FEM_noise(double beta,double factor,int cut,int** Condi,int var_index);
void set_solution_to_zero(double* Solution,int i_var,double (*F)(double x,double y));

double get_last_time(FILE* File,int* Mesh_size,int* Varnum,int* Step_num);
double** get_last_step_data(FILE* File,int mesh_size,int varnum,int stepnum);
void read_mesh2D(char* path,char* Name,int quiet);
void create_linear_multi_matrix(int n,double t,double dt);
char** read_param_strings(char* path,char* name,int* Len);
//void free_multimatrix(int n);
void free_conditions();
//void choose_linear_matrix(int mesh_number,int i);
//void create_interpolation_map(int n);
void set_boundary_conditions();
//int** resctrict_conditions(int var,int varnum,int** conditions);
int** set_pure_conditions(int varnum,int cond);
//void set_zero_Neumann(sparse_matrix* A,double* F,int* Indices,double* Values,int len);
//void set_fixed(sparse_matrix* A,double* Solution,double* F,int** conditions);
inline void free_cond(int** Cond);
void vector_dirichlet(double* F,int size,int** conditions);
//void set_single_dirichlet(sparse_matrix* A,double* F,int var_index,int mesh_index,double value);
//int set_const_bound(int** Cond,int var,int size,double* Sol,
//	double* Old_Sol,double dt);
//void gauge(double* Sol,int var);
//double mean_gauge(double* Sol,int var);

#endif
