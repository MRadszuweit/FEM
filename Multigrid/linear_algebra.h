#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "File_stuff.h"
#include "red_black_tree.h"

#define AUTOMATIC -1

// Datentypen ///////////////////////////////

typedef struct SPARSE{
	double** Values;
	int** Indices;
	int *Len;
	int size;
}sparse_matrix;

typedef struct SPARSE3D{
	double** Values;
	int** Indices1;
	int** Indices2;
	int *Len;
	int size;
}sparse_matrix3D;

typedef struct tree_sparse_matrix{
	int* Map;
	int* Inverse_Map;
	int map_size;
	struct tree_sparse_matrix* Parent;
	struct tree_sparse_matrix** Childs;
	sparse_matrix* Lower_Cross;
	sparse_matrix* Upper_Cross;
	sparse_matrix* Leaf;
	int child_number;
}tree_sparse_matrix;

// Funktionen /////////////////////////////////////

void set_thread_num(int n);
void incomplete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void complete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void diagonal_preconditioner(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void test_ILU(double** A,double** L,double** U,int n);
//int gauss_seidel(double* Solution,sparse_matrix* A,double* b,int iterations,double tol);
int gauss_seidel(int (*Inv_NB_matvec)(int n,double* x,double* b),
	void (*NB_matvec)(int n,double* y,double* z),
	double* Solution,double* b,int size,int iterations,double tol);
void Gauss_elemination(sparse_matrix* A,double* b,double* Solution,double tol);
void LU_factorzation(double** A,double** L,double** U,int size);
void SOR(sparse_matrix* A,double* F,double* Sol,double alpha,int iter);
void sparse_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
sparse_matrix* get_fast_ILU_preconditioned_matrix(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U,double tol);
void LU_sparse_block_factorization(sparse_matrix* A,sparse_matrix* LB,sparse_matrix* UB,int* Map,int* Inv_Map);
void LU_sparse_block_solver(sparse_matrix* L,sparse_matrix* U,int* Map,int* Inv_Map,
 double* X,double* B);
void Multi_LU_sparse_block_factorization(sparse_matrix* A,sparse_matrix* LB,
 sparse_matrix* UB,int* Map,int* Inv_Map,int crit_size);

int* zero_int_list(int n);
int* clone_list(int* x,int n);
double* clone_vector(double* x,int n);
void copy_vector_to(double* x,double* y,int n);
void copy_vector_content(double* x,double* y,int x_pos,int y_pos,int size);
double vec_dist(double* x1,double* x2,int n);
void scalar_mult(double a,double* x,int n);
void vector_add(double* x,double* y,double factor,int n);
double* sparse_mult(sparse_matrix* A,double* x);
void linear_map(double* x,double fac,sparse_matrix* A,double* y);
void sparse_multiplication(sparse_matrix* A,double* x);
sparse_matrix* sparse_product(sparse_matrix* A,sparse_matrix* B);
double scalar(double* x1,double* x2,int n);
void sparse_add(sparse_matrix* A,sparse_matrix* B,double factor);
double row_times_row(sparse_matrix* A,sparse_matrix* B,int i_a,int i_b,int j_min,int j_max);
sparse_matrix* get_transpose(sparse_matrix* A);
double euklid_norm(double* x1,int n);
double* zero_vector(int n);
double** zero_matrix(int n,int m);
double matrix_norm(sparse_matrix* A);
void scalar_sparse_mult(double a,sparse_matrix* A);
double residuum(void (*Matvec)(int n,double* y,double* z),double* Sol,double* b,int n);
double matrix_residuum(sparse_matrix* A,double* Sol,double* b);
double* generate_vector(int size,double val);
void vector_normalize(double* x,int n);
void simple_pivot(sparse_matrix* A,double* B,int* Map);
void pivot(sparse_matrix* A,double* B,int* Map);
double* Lower_triangular_invert(double** L,double* X,int size);
double* Upper_triangular_invert(double** U,double* X,int size);
int is_symmetric(sparse_matrix* A,int i,int j);
double** get_inverse(double** Q,int n);
void make_nonzero_diag(sparse_matrix* A,double* F);
double get_worst_diagonal(sparse_matrix* A);
void vector_shift(double* x,int n,double a);
void partial_vector_shift(double* x,int start,int end,double a);
sparse_matrix3D* sparse_zero3D(int n);
sparse_matrix* get_dependence_pattern3D(sparse_matrix3D* A);
sparse_matrix* get_dependence_pattern(sparse_matrix* A);

int insert_sparse(sparse_matrix* A,double a,int i,int j);
void set_sparse(sparse_matrix* A,double a,int i,int j);
void sparse_in_sparse(sparse_matrix* A,sparse_matrix* B,int i_o,int j_o);
sparse_matrix* sparse_diagonal(double* d,int n);
sparse_matrix* sparse_zero(int length);
sparse_matrix* sparse_identity(int length);
sparse_matrix* restrict_matrix_rows(sparse_matrix* A,int* List,int list_size,int* index_map);
double* restrict_matrix_cols(sparse_matrix* A,double* Sol,int* index_map);
sparse_matrix* restrict_matrix(sparse_matrix* A,double* b,double** b_sub,int** index_map,
	double* Solution,double* Prev_solution,double tol);
double* restrict_vector(double* Vector,int* index_map,int n_sol,int n_res);
double* restrict_vector_by(int i_min,int i_max,double* Vector);
void map_back(double* Solution,double* Restricted,int n_sol,int* index_map);
int* generate_list(int* var_indices,int var_len,int grid_size,int whole_size);
double* get_resized_vector(double* Vector,int old_size,int new_size);
double* join_vectors(double* V1,int n1,double* V2,int n2);
void append_vector(double** V1,int n1,double* V2,int n2);
void insert_vector(double* Vector,double* vector,int start,int size);
double** convert_sparse_to_array(sparse_matrix* A,int col_size);
void convert_array_to_sparse(double** M_A,sparse_matrix* A,int col_size,int row_size);
void sparse_approximate(sparse_matrix* A,double tol);
void set_zero(double* Vector,int start,int end);
double* restrict_system(sparse_matrix* A,double* F,double* Sol,int start,int end);
void change_to_spatial_ordering(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int block_size);
void map_indices(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int* Map);
sparse_matrix* get_sub_sparse_matrix(sparse_matrix* A,int start,int end,int* Map,int* Inv_Map);
 
void Free_sparse(sparse_matrix* A); // makro free_sparse s.u.
void insert_element_at(sparse_matrix* A,double a,int i,int j,int pos);
void remove_element_at(sparse_matrix* A,int i,int pos);
void remove_element(sparse_matrix* A,int i,int j);
void reset_row(sparse_matrix* A,int i);
void delete_zeros(sparse_matrix* A);
sparse_matrix* clone(sparse_matrix* A);
void copy_content(sparse_matrix* A,sparse_matrix* B);
int find_element(sparse_matrix* A,int j);
int in_list(int* List,int size,int n);
int* get_nonzero_indices(double* Vector,int size,int* new_size);
double get_diag(sparse_matrix* A,int i);
void change_row(sparse_matrix* A,double* F,int i1,int i2);
void change_var_rows(sparse_matrix* A,double* F,int var1,int var2,int block_size);
void resize_matrix(sparse_matrix* A,int new_size);
sparse_matrix* get_resized_matrix(sparse_matrix* A,int mesh_size,int new_size,int equ_index,int var_index);
int get_position(int* list,int size,int val,int* Flag);
double get_matrix_element(sparse_matrix* A,int i,int j);
sparse_matrix* sparse_partial_identity(int length,int start,int end);
int diag_zero(sparse_matrix* A);
int sparse_is_isolated(sparse_matrix* A,int i);
sparse_matrix* make_positive_diag(sparse_matrix* A);
int square_matrix_clean(sparse_matrix* A);
void normalize_row_sum(sparse_matrix* A);
double sparse_trace(sparse_matrix* A);
double sparse_bilinear(double* y,sparse_matrix* A,double* x);
void L_triang_invert(sparse_matrix* L,double* x);
void U_triang_invert(sparse_matrix* U,double* x);
double ILU_residuum(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U,double* Sol,double* b);
double SOR_residuum(sparse_matrix* A,double* Sol,double* b,int alpha);
double degree_of_occupation(sparse_matrix* A);
int get_max_band_width(sparse_matrix* A);
int get_ave_band_width(sparse_matrix* A);
int L_triang_fast_invert(sparse_matrix* L,int** Ind,double** Val,int size,int I,int max_band_width,double tol);
int U_triang_fast_invert(sparse_matrix* U,int**Ind,double** Val,int size,int I,int max_band_width,double tol);
double get_diag_dominance(sparse_matrix* A);
double test_sparse_LU(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void sparse_row_permutation(sparse_matrix* A,int* Map);
void vector_permutation(double* F,int* Map,int n);
int is_L_triang_matrix(sparse_matrix* A);
int is_U_triang_matrix(sparse_matrix* A);

void print_table(double* V,char* name,int n);
void print_sparse(sparse_matrix* A);
void print_vector(double* V,int n);
void print_list(int* V,int n);
void print_2D_list(int** A,int n);
void print_sparse_table(sparse_matrix* A);
int test_sparse_matrix(sparse_matrix* A);
void test_LU(double** A,double** L,double** U,int size);
sparse_matrix* Lap_2D_Dirichlet(double a,double b,int dom_size);

double** create_2D_array(int n,int m);
double** free_2D_array(double** A,int n);
void print_2D_array(double** A,int n);
void print_sqr_array(double** A,int n,int m);
sparse_matrix* get_test_matrix();
sparse_matrix* get_1D_Laplace(double* F,int n);

double get_mean(double* List,int size);
double get_sqr_mean(double* List,int size);
void double_sort(int* List,double* Values,int size);
double get_partial_mean(double* List,int start,int end);

int* convert_to_UMFPACK_sizes(sparse_matrix* A);
int* convert_to_UMFPACK_indices(sparse_matrix* A,int N);
double* convert_to_UMFPACK_Values(sparse_matrix* A,int N);

#define free_sparse(A) Free_sparse(A);free(A)
#define free_sparse3D(A) Free_sparse3D(A);free(A)
#define resize_vector(v,old_size,new_size) v = (double*)realloc(v,new_size*sizeof(double));set_zero(v,old_size,new_size)

#endif
