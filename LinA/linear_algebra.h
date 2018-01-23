#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdarg.h>
#include "File_stuff.h"
#include "red_black_tree.h"
#include <mpfr.h>

#define AUTOMATIC -1
#define INIT NULL
#define NEG_DEFINITE -1.
#define POS_DEFINITE 1.

#define ASS_FLUX 1
#define ASS_DEFAULT 0 

// Datentypen ///////////////////////////////

typedef struct MATRIX_ELEMENT{
	double val;
	int* Ind;
}matrix_element;

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
	int* Len;
	int size;
}sparse_matrix3D;

typedef struct SPARSE_GENERAL{
	matrix_element* Elements;
	int size;
	int rank;
	int* Dimension;
}sparse_general;

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

typedef struct MAP_SPARSE{
	int* Indices;
	sparse_matrix* Pattern;
}map_sparse;

typedef struct PRODUCT_PATTERN{
	double**** Pairs;
	int** Indices;
	int** Size_2;
	int* Size_1;
	int Size_0;
}product_pattern;

typedef struct UMFPACK_MATRIX_INFO{
	int* Ind;
	int* Col;
	double* Val;
}umfpack_matrix_info;

typedef struct ASS_IFO{
	int active_size;
	int* Active;
	double* Signs;
	double* Constraint;
	int* Iter_history;
	int calls;
}ass_info;

// Funktionen /////////////////////////////////////

void set_thread_num(int n);
int incomplete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void complete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void diagonal_preconditioner(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
void test_ILU(double** A,double** L,double** U,int n);
//int gauss_seidel(double* Solution,sparse_matrix* A,double* b,int iterations,double tol);
int gauss_seidel(int (*Inv_NB_matvec)(int n,double* x,double* b),
	void (*NB_matvec)(int n,double* y,double* z),
	double* Solution,double* b,int size,int iterations,double tol);
void Gauss_elemination(sparse_matrix* A,double* b,double* Solution,double tol);
void LU_factorzation(double** A,double** L,double** U,int size);
int cholesky_factorization(double** A,double** C, int size);
int cholesky(sparse_matrix* A,sparse_matrix* C,sparse_matrix* CT);
void SOR(sparse_matrix* A,double* F,double* Sol,double alpha,int iter);
void sparse_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U);
sparse_matrix* get_fast_ILU_preconditioned_matrix(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U,double tol);
void LU_sparse_block_factorization(sparse_matrix* A,sparse_matrix* LB,sparse_matrix* UB,int* Map,int* Inv_Map);
void LU_sparse_block_solver(sparse_matrix* L,sparse_matrix* U,int* Map,int* Inv_Map,
 double* X,double* B);
void Multi_LU_sparse_block_factorization(sparse_matrix* A,sparse_matrix* LB,
 sparse_matrix* UB,int* Map,int* Inv_Map,int crit_size);
int power_method(sparse_matrix* A,double* max_eig,double* Max_vec,int max_iter,double eps,sparse_matrix* Original,double eps_neg);
int get_most_negative_eigenvector(sparse_matrix* A,double* eigenvalue, double* Eigenvector,int max_iter,double eps);
int get_neg_curvature_direction(sparse_matrix* A,double* curvature, double* Direction,int max_iter,double eps,double neg_eps);
int active_set_solver(sparse_matrix* A,double* Sol,double* b,sparse_matrix* B,double* L,double* U,int (*Solver)(sparse_matrix* A,double* X,double* b),
 int max_iter,double definiteness,int mode,ass_info** solver_info);
//void clean_active_solver();
void free_ass_info(ass_info** info);

int min(int a,int b);
int max(int a,int b);
int* zero_int_list(int n);
int* clone_list(int* x,int n);
double* clone_vector(double* x,int n);
void copy_vector_to(double* x,double* y,int n);
void copy_vector_content(double* x,double* y,int x_pos,int y_pos,int size);
double vec_dist(double* x1,double* x2,int n);
void scalar_mult(double a,double* x,int n);
void vector_add(double* x,double* y,double factor,int n);
double vector_contraction(double* x,int n);
double* sparse_mult(sparse_matrix* A,double* x);
void linear_map(double* x,double fac,sparse_matrix* A,double* y);
void sparse_multiplication(sparse_matrix* A,double* x);
sparse_matrix* sparse_product(sparse_matrix* A,sparse_matrix* B);
double scalar(double* x1,double* x2,int n);
long double hp_scalar(double* x1,double* x2,int n);
void sparse_add(sparse_matrix* A,sparse_matrix* B,double factor);
double row_times_row(sparse_matrix* A,sparse_matrix* B,int i_a,int i_b,int j_min,int j_max);
sparse_matrix* get_transpose(sparse_matrix* A,int new_size);
double euklid_norm(double* x1,int n);
long double hp_euklid_norm(double* x1,int n);
double sqr_euklid_norm(double* x,int n);
double* zero_vector(int n);
double** zero_matrix(int n,int m);
double matrix_norm(sparse_matrix* A);
double get_largest_matrix_element(sparse_matrix* A);
double estimate_operator_norm(sparse_matrix* A);
void scalar_sparse_mult(double a,sparse_matrix* A);
double residuum(void (*Matvec)(int n,double* y,double* z),double* Sol,double* b,int n);
double matrix_residuum(sparse_matrix* A,double* Sol,double* b);
double* generate_vector(int size,double val);
void vector_normalize(double* x,int n);
void simple_pivot(sparse_matrix* A,double* B,int* Map);
void other_pivot(sparse_matrix* A,double* B,int* Map);
void pivot(sparse_matrix* A,double* B,int* Map);
int* get_inverse_index_map(int* Map,int size);
double* Lower_triangular_invert(double** L,double* X,int size);
double* Upper_triangular_invert(double** U,double* X,int size);
int is_symmetric(sparse_matrix* A,int i,int j);
double quadratic_form(sparse_matrix* A,double* b,double* x);
double** get_inverse(double** Q,int n);
void make_nonzero_diag(sparse_matrix* A,double* F);
double get_worst_diagonal(sparse_matrix* A);
void vector_shift(double* x,int n,double a);
void partial_vector_shift(double* x,int start,int end,double a);
sparse_matrix3D* sparse_zero3D(int n);
sparse_matrix* get_dependence_pattern3D(sparse_matrix3D* A);
sparse_matrix* get_dependence_pattern(sparse_matrix* A);
void manuel_pivot(sparse_matrix* A,double* B,int* Map,int* I,int* J,int ind_size,int block_size);
int* get_ID_index_map(int size);
void vector_pseudo_mult(double* x,double* y,int n);
void vector_pseudo_div(double* x,double* y,int n);
double* vector_join(double* x,double* y,int size_x,int size_y);
sparse_matrix* matrix_pseudo_left_mult(double* X,sparse_matrix* A,int dim);
sparse_matrix* matrix_pseudo_right_mult(double* X,sparse_matrix* A,int dim);
void sparse_pseudo_left_mult(double* X,sparse_matrix* A,int dim);
void sparse_pseudo_right_mult(double* X,sparse_matrix* A,int dim);
void set_random_normal_vector(double* X,int n);
int symmetry(sparse_matrix* A,double tol);
sparse_matrix* sparse3D_vB(sparse_matrix3D* B,double* X,int new_size);
double* sparse3D_Bvv(sparse_matrix3D* B,double* X1,double* X2);
double* sparse3D_vvB(sparse_matrix3D* B,double* X1,double* X2,int size);
void sparse3D_scalar_mult(sparse_matrix3D* B,double a);
void sparse3D_add(sparse_matrix3D* A,sparse_matrix3D* B);
double* sparse3D_bilinear(sparse_matrix3D* B,double* X1,double* X2);
sparse_matrix* sparse3D_Bv(sparse_matrix3D* B,double* X);
sparse_matrix* sparse3D_Bv_symmetric(sparse_matrix3D* B,double* X);
sparse_matrix* sparse3D_Bv_middle(sparse_matrix3D* B,double* X);
int sparse3D_element_number(sparse_matrix3D* B);
void sparse3D_merge(sparse_matrix3D* B);
sparse_matrix3D* get_transpose3D_12(sparse_matrix3D* B,int new_size);
sparse_matrix3D* get_transpose3D_13(sparse_matrix3D* B,int new_size);
sparse_matrix3D* sparse3D_transpose13(sparse_matrix3D* B,int new_size);


int insert_sparse(sparse_matrix* A,double a,int i,int j);
void insert_row(sparse_matrix* A,double* R,int i);
void set_sparse(sparse_matrix* A,double a,int i,int j);
void sparse_in_sparse(sparse_matrix* A,sparse_matrix* B,int i_o,int j_o);
void add_subsparse_to_sparse(sparse_matrix* A,sparse_matrix* B,int i_o,int j_o);
sparse_matrix* sparse_diagonal(double* d,int n);
sparse_matrix* sparse_zero(int length);
sparse_matrix* sparse_identity(int length);
void remove_zero(sparse_matrix* A,int i);
void remove_zeros(sparse_matrix* A);
sparse_matrix* restrict_matrix_rows(sparse_matrix* A,int* List,int list_size,int* index_map);
double* restrict_matrix_cols(sparse_matrix* A,double* Sol,int* index_map);
sparse_matrix* restrict_matrix(sparse_matrix* A,double* b,double** b_sub,int** index_map,
	double* Solution,double* Prev_solution,double tol);
void enlarge_matrix(sparse_matrix* A,int mesh_size,int new_size,int equ_index,int var_index);
void simple_enlarge_matrix(sparse_matrix* A,int add_size);
double* restrict_vector(double* Vector,int* index_map,int n_sol,int n_res);
double* restrict_vector_by(int i_min,int i_max,double* Vector);
void map_back(double* Solution,double* Restricted,int n_sol,int* index_map);
int* generate_list(int* var_indices,int var_len,int grid_size,int whole_size);
double* get_resized_vector(double* Vector,int old_size,int new_size);
double* join_vectors(double* V1,int n1,double* V2,int n2);
void append_vector(double** V1,int n1,double* V2,int n2);
void insert_vector(double* Vector,double* vector,int start,int size);
double** convert_sparse_to_array(sparse_matrix* A,int col_size);
void convert_transposed_array_to_sparse(double** M_A,sparse_matrix* A,int col_size,int row_size);
void convert_array_to_sparse(double** M_A,sparse_matrix* A,int col_size,int row_size);
void sparse_approximate(sparse_matrix* A,double tol);
void set_zero(double* Vector,int start,int end);
double* restrict_system(sparse_matrix* A,double* F,double* Sol,int start,int end);
void change_to_spatial_ordering(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int block_size);
void map_indices(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int* Map);
sparse_matrix* get_sub_sparse_matrix(sparse_matrix* A,int start,int end,int* Map,int* Inv_Map);
void copy_sparse_content(sparse_matrix* A,sparse_matrix* B);
void sparse_left_mult(sparse_matrix* A, sparse_matrix* B);
void sparse_right_mult(sparse_matrix* A, sparse_matrix* B);
double* sparse_row_sum(sparse_matrix* A);
int get_max_col_index(sparse_matrix* A);
sparse_matrix* get_inv_U_triang(sparse_matrix* U);
sparse_matrix* serial_sparse_product(sparse_matrix* A,sparse_matrix* B);
double* get_row_sums(sparse_matrix* A);
double* get_col_sums(sparse_matrix* A,int col_num);
sparse_matrix* get_sparse_part(sparse_matrix* A,int start,int end);
double* get_inv_diag(sparse_matrix* A);
void normalize_col_sum(sparse_matrix* A,double* Col_Sums,int col_num);
void sub_sparse_insert(sparse_matrix* A,sparse_matrix* Small,int equ,int var,int d);
int find_position_in_row(sparse_matrix* A,int row,int col);
void free_sparse_index_map(map_sparse** Map);
 
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
void normalize_row_sum(sparse_matrix* A,double* Row_Sums);
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
void sparse_get_optimized_index_map(sparse_matrix* A,int* Map,int* Inv_Map);
product_pattern set_product_pattern(sparse_matrix* A,sparse_matrix* B,sparse_matrix* B_transposed,sparse_matrix** AB);
void free_product_pattern(product_pattern* P);
void fast_sparse_product(sparse_matrix* P, sparse_matrix* A,sparse_matrix* BT,product_pattern* Pattern);
int get_max_mem_bandwidth(sparse_matrix* A);
sparse_matrix* cols_to_sparse(double** V,int col_num,int v_size);
sparse_matrix* rows_to_sparse(double** V,int row_num,int v_size);
void apply_func(double* X,int size,double (*Func)(double x));

void print_table(double* V,char* name,int n);
void print_sparse(sparse_matrix* A);
void print_3D_sparse(sparse_matrix3D* A);
void print_vector(double* V,int n);
void print_list(int* V,int n);
void print_2D_list(int** A,int nx,int ny);
void print_scalar_data(char* Name,double* V,int n);
void print_sparse_table(sparse_matrix* A);
void print_to_matrix_market_format(sparse_matrix* A,char* Name);
int test_sparse_matrix(sparse_matrix* A);
void test_LU(double** A,double** L,double** U,int size);
sparse_matrix* Lap_2D_Dirichlet(double a,double b,int dom_size);
sparse_matrix* read_sparse_textfile(char* path,char* name);
void write_sparse_textfile(char* path,char* name,sparse_matrix* A);
int write_sparse(sparse_matrix* A,char* path,char* name);
sparse_matrix* read_sparse(char* path,char* name);
int test_vector(double* X,int n,char* message);
int test_negative(double* X,int n);
void remove_negative(double* X,int n);
void print_matrix_diff(char* Name1,char* Name2,double tol);

double** create_2D_array(int n,int m);
double** free_2D_array(double** A,int n);
void print_2D_array(double** A,int n);
void print_sqr_array(double** A,int n,int m);
sparse_matrix* get_test_matrix(int n,int random_rot);
sparse_matrix* get_1D_Laplace(double* F,int n);
double* rank4_product(double* A,double* B,int dim,int mesh_size);
double* rank4_tensor_product(double* A,double* B,double* C,double* D,int dim,int mesh_size);
void add_rank4_tensor_product(double* X,double* A,double* B,double* C,double* D,int dim,int mesh_size);

double get_mean(double* List,int size);
double get_sqr_mean(double* List,int size);
void int_sort(int* List,int size);
void double_sort(int* List,double* Values,int size);
double get_partial_mean(double* List,int start,int end);
double get_max_element_abs(double* X,int size);
double get_min_element_abs(double* X,int size);
void divide_and_conquer_sort(int* List,void** Values,int size,int (*Compare)(void* X1,void* X2));
int DC_get_position(void* Element,void** Values,int size,int* Flag,int (*Compare)(void* X1,void* X2));
int cmp_multidim_index(void* X1,void* X2);
void join_lists(int** List,int* size,int* L1,int* L2,int s1,int s2);

void test_omp(int size);
void set_omp_chunk_size(int chunk);

/*sparse_general* sparse_general_contraction(sparse_general* C,double** X,int* Contr_index,int num_fields);
sparse_matrix* sparse_general_to_matrix(sparse_general* C,int col_size);
double* sparse_general_to_vector(sparse_general* C,double vec_size);*/

sparse_general* new_sparse_general(int rank,int* Dim);
void free_sparse_general(sparse_general** A);
void sparse_general_append(sparse_general* A,double val,int rank,...);
void sparse_general_Append(sparse_general* A,double val,int rank,int* Ind);
void sparse_general_add(sparse_general* A,sparse_general* B,double factor,int sorted);
void sparse_general_scalar_mult(sparse_general* A,double factor);
double sparse_general_to_scalar(sparse_general* A,int arg_num,...);
double* sparse_general_to_vector(sparse_general* A,int arg_num,...);
sparse_matrix* sparse_general_to_sparse2D(sparse_general* A,int arg_num,...);
sparse_general* sparse_general_contraction(sparse_general* A,int arg_num,...);
sparse_general* sparse3D_to_general(sparse_matrix3D* B,int dim1,int dim2,int dim3);
void sparse_general_sort(sparse_general** A);
double general_matrix_norm(sparse_general* A);
void print_sparse_general(sparse_general* A);

void perform_givens_rotation(sparse_matrix* A,int i,int j,double c,double s);
void Hessenberg_QR(sparse_matrix* H,sparse_matrix** Tri,sparse_matrix** Q);
void QR_decomposition(sparse_matrix* A,sparse_matrix** R,sparse_matrix** Q);
void Arnoldi(sparse_matrix* A,sparse_matrix** H,double** U,double* V0,int* s);
void reortho_Arnoldi(sparse_matrix* A,sparse_matrix** H,double** U,double* V0,int* s,int iter);
void Ritz_values(sparse_matrix* H,double** V,double* Eig,int iter);

void convert_sparse_to_UMFPACK(sparse_matrix* A,int col_num,int** Col_start,int** Indices,double** Values,int* Size);
void convert_UMFPACK_to_sparse(sparse_matrix** A,int col_num,int row_num,int* Col_start,int* Indices,double* Values);
void free_UMFPACK_matrix_info(umfpack_matrix_info* A);
//void convert_sparse_to_UMFPACK_long(sparse_matrix* A,int col_num,SuiteSparse_long** Col_start,SuiteSparse_long** Indices,double** Values,int* Size);


#define free_sparse(A) Free_sparse(A);free(A)
#define free_sparse3D(A) Free_sparse3D(A);free(A)
#define resize_vector(v,old_size,new_size) v = (double*)realloc(v,new_size*sizeof(double));set_zero(v,old_size,new_size)

#endif
