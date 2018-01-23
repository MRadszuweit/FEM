 /* lib datenstrukturen */

#ifndef FEM2D_H
#define FEM2D_H
  
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <execinfo.h>
#include <errno.h>
#include "geometry2D.h"
#include "linear_algebra.h"
#include "red_black_tree.h"
//#include <gts.h>

#define bool short
#define true 1
#define false 0
#define UP 1
#define DOWN -1
#define INVALID 1E100
#define SIN 1
#define COS 0
#define PARTITION_ON 1
#define PARTITION_OFF 0
#define NOCON 0
#define DIRICHLET 1
#define NEUMANN 2
#define ROBIN 3
#define MIXED 4
 
// Datentypen ////////////////////////////////////////////

 typedef struct INDEX2D{
 	int i;
 	int j;
}index2D; 

typedef struct INDEX3D{
 	int i;
 	int j;
 	int k;
 }index3D;
 
 typedef struct ELEMENT_COLLECTION{
	 index3D* Elements;
	 int size;
 }element_collection;
 
 typedef struct MESH2D{
 	point2D* Points;			//! Gitterpunkte
 	int** Connections;			//! Listen mit den Nachbarindizes
 	int* Sizes;				//! Anzahl der Nachbarn 
	int* Is_boundary;			//! ist Randpunkt
 	int size;				//! Anzahl der Gitterpunkte
	int*** Overlap_table;			//! Look-up-table für overlap-function
 }mesh2D;
 
 typedef struct INTERPOLATION_MAP2D{
    index3D*** Elements;			//!Dreiecksindices des i-ten Gitters
	int* Triangle_numbers;			//! Anzahl der Dreiecke des i-ten Gitters
	int* Vertex_numbers;			//! Anzahl der Vertices des i-ten Gitters
	int** Map_up;					//! Zuordnung der Vertices i zu den Dreiecken des niedrigeren Gitters i-1
	int** Map_down;					//! Zuordnung der Vertices i zu den Dreiecken des höheren Gitters i+1
 }interpolation_map2D;
 
 typedef struct SLAB_TREE{
	 rb_red_blk_tree* Tree;
	 double x;
	 double y;
 }slab_tree;
 
 typedef struct SEARCH_TREE2D{
	 rb_red_blk_tree* Main_tree;
	 mesh2D* Mesh;
	 element_collection* Triangles;
	 int last_index;
 }search_tree2D;
 
 typedef struct ISOLINE{
	 index3D* Tri_indices;				//! index of the triangle element
	 point2D* Coordinates;			//! barycentric coordinates a,b (1-a-b), where a refers to ind.i and b to ind.j
	 int size;						//! length of isoline 
 }isoline;
 
// Funktionen ////////////////////////////////////////////

void init_mesh(int size);
void test_mesh();
void free_mesh();
void free_elements();
void free_multimeshes();
void free_interpolation_map2D();

double det(point2D* P1,point2D* P2,point2D* P3);
point2D grad_2D(int i,int j,int k);
point2D bound_line_2D(int i,int j,int k);
double det_tri(int i,int j,int k);
double get_longest_edge(mesh2D* Mesh);
int is_boundary(int i);
int IsBoundary(mesh2D* Mesh,int i);
int* overlap(int i,int j);
int* Overlap(mesh2D* Mesh,int i,int j);
int near_boundary(int i);
void get_shortest_boundary_path(int start,int end,int**List,int* len);
void sortknot(int i);
void sort_knots();
void set_mesh(int n);
void set_mesh_number(int n);

double vector_b_0(int i);
double vector_Ui_0(int i);
double* set_vector_Ui_0_2D(int var_num);
double* Get_vector_Ui_0_2D(mesh2D* Mesh);

////////////////// second order 

double matrix_Aii_00(int i);
double matrix_Aij_00(int i,int j);
point2D matrix_Aii_01(int i);
point2D matrix_Aij_01(int i,int j);
matrix2D matrix_Aii_11(int i);
matrix2D matrix_Aij_11(int i,int j);

////////////////// third order

point2D matrix_Biii_001(int i);
point2D matrix_Biij_001(int i,int j);
point2D matrix_Biji_001(int i,int j);
point2D matrix_Bijj_001(int i,int j);
point2D matrix_Bijk_001(int i,int j,int k);

matrix2D matrix_Biii_011(int i);
matrix2D matrix_Biji_011(int i,int j);
matrix2D matrix_Bijj_011(int i,int j);
matrix2D matrix_Biij_011(int i,int j);
matrix2D matrix_Bijk_011(int i,int j,int k);

//////////////////// Matrixprodukte

void insert_A_aV_a(sparse_matrix* A,point2D* v,int i,int j,int d);

void insert_A_abV_b(sparse_matrix* A,matrix2D* v,int i,int j,int d);
void insert_A_baV_b(sparse_matrix* A,matrix2D* v,int i,int j,int d);
void insert_A_bbV_a(sparse_matrix* A,matrix2D* v,int i,int j,int d);

void insert_A_abV(sparse_matrix* A,matrix2D* v,int i,int j,int d);
void insert_A_bbV(sparse_matrix* A,matrix2D* v,int i,int j,int d);
void insert_A_aV(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_aV_b_row_x(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_aV_b_row_y(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_bV_a_row_x(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_bV_a_row_y(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_cV_cE_ab_row_x(sparse_matrix* A,point2D* v,int i,int j,int d);
void insert_A_cV_cE_ab_row_y(sparse_matrix* A,point2D* v,int i,int j,int d);

void insert_A_bV_ba(sparse_matrix* A,point2D* v,int i,int j,int d);

void insert_pseudo_A_aV_a(sparse_matrix* A,point2D* v,int i,int j,int d);

void insert_AV(sparse_matrix* A,double v,int i,int j,int d);
void insert_AV_a(sparse_matrix* A,double v,int i,int j,int d);

void insert_BWWW(sparse_matrix3D* B,double v,int i,int j,int k,int d);

void insert_B_aWWW(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);

void insert_BWWW_a(sparse_matrix3D* B,double v,int i,int j,int k,int d);

void insert_BWV_aV_a(sparse_matrix3D* B,double v,int i,int j,int k,int d);

void insert_B_aWWV_a(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_bV_bV_a(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_aaV_bV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_V_aB_abV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_V_aB_baV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);

void insert_B_aaWWW(sparse_matrix3D* b,matrix2D* v,int i,int j,int k,int d);
void insert_B_aaV_bV_bW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_B_abV_bV_aW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_B_baV_bV_aW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);

void insert_UBa_V_a(sparse_matrix3D*B,point2D* v,int i,int j,int k,int d);
void insert_B_aWV_b_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_aWV_b_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_bWV_a_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_bWV_a_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_cWV_cEab_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_cWV_cEab_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_abV_cV_c_rowx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_B_abV_cV_c_rowy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_WB_xx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_WB_xy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_WB_yx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_WB_yy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);

void insert_B_divergence(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);
void insert_B_gradient(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d);


/*void insert_BV_aW(sparse_matrix* A,double v,double* data,int i,int j,int k,int d);
void insert_B_aV_aW(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d);
void insert_B_aVW(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d);
void insert_B_aVW_a(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d);

void insert_V_aB_baV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_V_aB_abV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);
void insert_B_aaV_bV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d);*/

void insert_B(double* B,double v,int i,int d);
void insert_Ba(double* B,double v,int i,int d);

/*void insert_WV_abV_ba(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d);
void insert_WV_abV_ab(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d);
void insert_WV_aaV_bb(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d);*/

void insert_WWV_aC_baV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d);
void insert_WWV_aC_abV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d);
void insert_WWC_aaV_bV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d);

matrix2D CB(int i);

//////////////////// matrices

// second order
sparse_matrix* set_matrix_Aij_00_2D(int equ,int var,
	void (*insert)(sparse_matrix* A,double v,int i,int j,int d));
sparse_matrix* set_matrix_Aij_01_2D(int equ,int var,
    void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d));
sparse_matrix* set_matrix_Aij_10_2D(int equ,int var,
    void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d));
sparse_matrix* set_matrix_Aij_11_2D(int equ,int var,
    void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d));

sparse_matrix* set_matrix_bound_0_Aij_00_2D(int equ,int var,
	void (*insert)(sparse_matrix* A,double v,int i,int j,int d));
	
sparse_matrix* set_matrix_bound_1_Aij_00_2D(int equ,int var,
	void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d));

sparse_matrix* set_matrix_bound_2_Aij_10_2D(int equ,int var, 
	void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d));

sparse_matrix* get_matrix_Aij_00_2D(mesh2D* Mesh,int equ,int var,
void (*insert)(sparse_matrix* A,double v,int i,int j,int d));
    
// third order

sparse_matrix3D* set_matrix_Bijk_000(int equ,int var1,int var2,				
  void (*insert)(sparse_matrix3D* B,double v,int i,int j,int k,int d));

sparse_matrix3D* set_matrix_Bijk_001(int equ,int var1,int var2,
	void (*insert)(sparse_matrix3D* A,point2D* v,int i,int j,int k,int d));
	
sparse_matrix3D* set_matrix_Bijk_011(int equ,int var1,int var2,
	void (*insert)(sparse_matrix3D* A,matrix2D* v,int i,int j,int k,int d));

sparse_matrix3D* set_matrix_bound_2_Bijk_100_2D(int equ,int var1,int var2,
	void (*insert)(sparse_matrix3D* A,matrix2D* v,int i,int j,int k,int d));

// fourth order

sparse_general* set_matrix_Cijkl_0011_2D(int var1,int var2,int var3,int var4,
	void (*insert)(sparse_general* V,matrix2D* v,int i,int j,int k,int l,int d));

//////////////////// other stuff

double* set_vector_bi_0(int var,void (*insert)(double* B,double v,int i,int d));

double* set_power_0(double (*func)(point2D* P),int equ);
double* set_power_1(double* Solution,int equ,int var,double factor);
//double* set_power_2(double* Solution,int equ,int var,double factor);
//double* set_power_3(double* Solution,int equ,int var,double factor);

double* glob_div(int var);
double* glob_rot(int var);
//double* hesse(double* Data);

void Null_mesh(mesh2D* Mesh);
double mean2D(double* X);
double variance2D(double* X);
void FreeMesh(mesh2D* Mesh);
void Sort_knots(mesh2D* Mesh);
sparse_matrix* set_matrix_unity(int equ,int var);
sparse_matrix* stretch_matrix(double ax,double ay,int var_num,int var_ind);
sparse_matrix* get_diag_matrix_function(int var_num,double (*F)(int mesh_ind,int var_ind));
double Tschebyscheff_disc(int i,int n,int m,int trig);
bool in_array(int i,int j,int n);
double* cartesian_function_on_mesh2D(double (*Func)(double x,double y));
int index_conversion(int i,int j,int n);
index2D inv_index_conversion(int k,int n);
void set_var_number2D(int n);
int get_var_number2D();
void pyramid();
int get_center_index();
void get_boundary_elements(int** Indices,int* len);
void get_mesh_bounds2D(point2D* Min,point2D* Max,mesh2D* Mesh,double tol);
void create_interpolation_map2D(int mesh_index,int mode);
double linear_interpolation2D(point2D* P1,point2D* P2,point2D* P3
	,double u1,double u2,double u3,point2D* P);
point2D triangle_center(point2D* P1,point2D* P2,point2D* P3);
int Inside_triangle(mesh2D* Mesh,element_collection* Triangles,point2D* P,int index,double tol);
int Inside_triangle_set(mesh2D* Mesh,element_collection* Triangles,point2D* P,int* List,double tol);
void seek_whole(int mesh_index,int ind,int mode);
void find_nearest_triangle2D(int mesh_index,int ind,int mode);
void interpolation2D(double* Dest,double* Source,int mesh_index,int varnum,int mode);
double interpolate_single_value(double* X,point2D* P);
void test_interpol(int mesh_index,int mesh_division);
void create_look_up_table(mesh2D* Mesh);
double get_total_area();
double get_total_amount(double* X,int index);
double max_edge_len();
double mean_edge_len();
int get_boundary_edge_num(mesh2D* Mesh);
int IsBoundaryEdge(mesh2D* Mesh,index2D ind);
double bound_curvature(int i);
int get_nearest_index(point2D* P);
int* partition2D(mesh2D* Mesh,int* Borders,point2D* Seeds,int n_border,int n_seed);
void set_partition_mode(int mode);
void init_partition(int* List,double* Attr,int size);
void set_part_attr(double* Attr);
double get_attr(int i);
sparse_matrix* set_attr_matrix(int equ,int var,int varnum);
double get_bound_curvature(int i);
int* get_partition();
void enable_partition();
void disable_partition();
point2D* create_regular_mesh2D(int n,int m,double rim_size,point2D* Ref);
void create_tri_map(point2D* New_nodes,int** Tri_map,int** Inv_tri_map,int size,double r);
void mesh_interpolation(point2D* New_nodes,double* New_values,int size,double* Values,int** Tri_map);
void inverse_interpolation(point2D* New_nodes,double* New_values,double* Values,int** Inv_tri_map);
point2D get_reg_dist(int x_size,int y_size,double rim_size);
void refine_mesh(char* filename,double (*area_function)(point2D* P),int quiet);
void Refine_mesh(char* filename,double* Area_data,int size,int quiet);
double* generate_2D_vector(point2D* V,int mesh_size);
index2D get_most_distinct_points();
int Get_nearest_index(mesh2D* Mesh,point2D* P);
sparse_matrix* get_stiffness_tensor(double* Rank4_coeff);
void edge_refine(point2D** Poly,int* Loop_start,int loop_num,double threshold,double dmin,double edge_refine_fraction);
double get_largest_element_area(mesh2D* Mesh,element_collection* Triangles);
void add_constraint_points(char* FileIn,char* FileOut,point2D* Points,int size);
sparse_matrix* get_convolution_matrix(double (*Kernel)(point2D* P1,point2D* P2),double range);
double moment_2D(double* X,int order);
void set_linear_coeffcients_2D(point2D* A,point2D* B,point2D* C,double* a,double* b,double* c);
int* get_boundary_conditions(point2D** ChangeB,int** Cond,int* Sizes,int loops);
double* spread_values2D(mesh2D* Mesh,int** Polygons,int* Sizes,point2D* Seeds,double* Values,int poly_num,int attr_num);
double* Field_histogramm(double* X,int resolution);
void print_histogram(char* Filename,double* Hist,int size);
int in_polygon(int* Poly,int size,point2D* P);
void print_area_weighted_vector(char* Name,double* Data);
rb_red_blk_node* red_blk_get_largest_left_node(rb_red_blk_tree* tree,rb_red_blk_node* node,void* key);
//rb_red_blk_tree* slab_search_tree(mesh2D* Mesh,element_collection* Triangles,int* end_tri_index);
search_tree2D* create_search_tree2D(mesh2D* Mesh,element_collection* Triangles);
//slab_tree* get_left_slab_tree(rb_red_blk_tree* Main_tree,point2D* P);
int get_triangle_index(search_tree2D* Search_tree,point2D* P);
void free_search_tree2D(search_tree2D** Tree);
int get_nearest_triangle(mesh2D* Mesh,element_collection* Triangles,point2D* P);
int** get_triangle_list(mesh2D* Mesh,element_collection* Triangles);
point2D get_barycentric_2D(point2D* A,point2D* B,point2D* C,point2D* P);
double* tri_to_vert(mesh2D* Mesh,element_collection* Triangles,double* Data);
isoline* get_isoline(mesh2D* Mesh,double* X,int start_index,double tol,int max_len);
void isoline_to_vertices(mesh2D* Mesh,isoline* Iso,point2D** Vertices,int* size);
void find_iso_start_indices_eq_dist(double* X,point2D* P1,point2D* P2,int num,int* Indices);
void free_isoline(isoline** Iso);

//point2D* get_voronoi_edges(int k,double* Shift);
//double get_voronoi_area(int k,double* Shift);
//point2D* get_flux_coefficients(int k,double* Shift);

point2D* flux_weights(int k);
double* val_weights(int k);
sparse_matrix* FVM_convection_matrix2D(double* Flow_field,double* Shift_field,sparse_matrix* Coeff,double dt);
sparse_matrix* FVM_convective_fluxes_2D(double* Density,double* Flow_field,double* Shift_field);
void FVM_convection_2D(double* Density_field,double* Flow_field,double* Shift_field,double dt);
void FEM2D_get_optimized_index_map(int* Map,int* Inv_Map,int varnum);
//sparse_matrix* get_barycentric_interpolation_matrix(mesh2D* Dest,mesh2D* Source,element_collection* Source_elements);
sparse_matrix* FEM2D_interpolation_matrix(mesh2D* Dest,mesh2D* Source,element_collection* Dest_elements,
 element_collection* Source_elements,double d,double tol);
//double* get_minimum_element_interpolation(double* X,sparse_matrix* Int_matrix,mesh2D* Dest,element_collection* Dest_elements,mesh2D* Source,element_collection* Source_elements);
//double* FEM_get_minimum_interpolation(double* X,mesh2D* Dest,element_collection* Dest_elements,mesh2D* Source,double d);
double max_triangle_size(mesh2D* Mesh,element_collection* Triangles);
sparse_matrix* sparse3D_times_rank4_2D_left(sparse_matrix3D* B,double* C);
sparse_matrix* sparse3D_times_rank4_2D_right(sparse_matrix3D* B,double* C);
sparse_matrix* sparse3D_times_rank4_2D_right_T(sparse_matrix3D* B,double* C);
void sparse3D_times_rank4_2D_right_get_map(sparse_matrix3D* B,sparse_matrix* A,map_sparse** Map);
sparse_matrix* sparse3D_times_rank4_2D_right_use_map(sparse_matrix3D* B,double* C,map_sparse* Map);
sparse_matrix* sparse3D_times_rank4_2D_middle(sparse_matrix3D* B,double* C,double* U);
double* sparse3D_times_rank4_2D_bilinear(sparse_matrix3D* B,double* C,double* X1,double* X2);
double* sparse3D_times_rank4_matrix_matrix(sparse_matrix3D* B,double* M,double* P,double* C);
double* sparse3D_times_rank4_matrix_vector(sparse_matrix3D* B,double* C,double* M,double* X);
double* sparse3D_times_rank4_matrix(sparse_matrix3D* B,double* C,double* M);
double* rank4_unity_2D();
double* rank4_2D_first_trace(double* C);
double* rank2_times_rank4_2D(double* M2,double* C4);
sparse_matrix* sparse3D_times_rank2_2D_left(sparse_matrix3D* B,double* G);
sparse_matrix* sparse3D_times_rank2_2D_right(sparse_matrix3D* B,double* G);
double* sparse3D_times_rank2_times_scalar_2D_right(sparse_matrix3D* B,double* Gab,double* X);
double* sparse3D_bilinear_tensor_2D_right(sparse_matrix3D* B,double* X,double* Y);
double* nodewise_tensor_contract_2D(double* A,double* B);

void feed_line_int();

#endif
