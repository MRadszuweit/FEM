/* lib: geometry */

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Datentypen /////////////////////////////////////

typedef struct POINT2D{
 	double x;
 	double y;
 }point2D;
 
typedef struct MATRIX2D{
 	double xx;
 	double xy;
	double yx;
	double yy;
 }matrix2D;
 
// Funktionen /////////////////////////////////////

point2D* clone_point(point2D* P);
point2D init_point2D(double x,double y);
void delete_duplicate_points(point2D** List,int* size,double tol);

void vec_mult(point2D* v,double a);
point2D vec_diff(point2D* a,point2D* b);
void vec_add_mult(point2D* a,point2D* b,double fac);
double vec_scalar(point2D* a,point2D* b);
double vec_abs(point2D* v);
double Vec_abs(point2D v);
double vec_cross2D(point2D* a,point2D* b);

double determinant_2D(matrix2D* A);
matrix2D* get_inverse_transposed_2D(matrix2D* A);
matrix2D* get_affine_2D(point2D* p1,point2D* p2,point2D* p3);
void get_affine_info_2D(point2D* p1,point2D* p2,point2D* p3,matrix2D* F,matrix2D* IFT,double* detF);

void mat_vec_mult(matrix2D* A,point2D* P);
void mat_mat_mult(matrix2D* A,matrix2D* B);
void mat_mult(matrix2D* A,double a);
void mat_add(matrix2D* A,matrix2D* B,double a);
matrix2D* unity_matrix2D();
matrix2D transpose(matrix2D A);
matrix2D* get_transpose_2D(matrix2D* A);
matrix2D* tensor_product(point2D* v1,point2D* v2);
double trace_2D(matrix2D* A);
double contraction(matrix2D* A,matrix2D* B);

double dist(point2D* a,point2D* b);
double sqrdist(point2D* a,point2D* b);
void normalize(point2D* v);
point2D* get_normalized(point2D* v);
point2D Get_normal(point2D p1,point2D p2,point2D inner);
point2D get_normal(point2D* P1,point2D* P2,point2D* inner);
point2D ortho_projection(point2D projector,point2D v);
//double* rank4_projector_2D(double nx,double ny);
double* rank4_trace_part_2D(double* C);

int in_triangle(point2D* A,point2D* B,point2D* C,point2D* P);
double polygon_area(point2D* Edges,int size);
double Polygon_area(point2D** Poly,int size);
double Polygon_Area(point2D* Points,int* Poly,int size);
double Area_form2D(point2D* A,point2D* B,point2D* C);
void get_affine_trafo_2D(point2D* P1,point2D* P2,point2D* P3,double** F,double** G);
int congruent_triangle(int i1,int j1,int k1,int i2, int j2,int k2);

void point2D_sort(int* List,point2D** Values,int size,point2D* Ref);

void print_point_list2D(point2D* V,int n,char* Name);
double create_arc(int n,double r,double angle,point2D* data);
void create_poly_file(char* name,point2D* data,int total_size);
void create_nonconvex_poly_file(char* name,point2D** data,int* Sizes,int number,point2D* Outer);
void create_attribute_poly_file(char* name,point2D** data,int* Sizes,int number,point2D* Inner,double* Attr);
void polygons_from_polyfile(char* name,point2D* Points,int*** Polys,int** Sizes,point2D** Seeds,double** Values,int* number,int* attr_num,int quiet);

int chance_to_overlap(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2);
point2D* line_intersection_2D(point2D* P1,point2D* P2,point2D* Q1,point2D* Q2,double tol);
point2D* line_intersection_positive_2D(point2D* P1,point2D* P2,point2D* Q1,point2D* Q2,double tol);
void triangle_intersection_2D(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,point2D*** Poly_list,int* poly_size,double tol);
double* barycentric_iso_line(double a, double b, double c,point2D* Grad_ab,double tol);

void print_mathematica_polygon(point2D** Poly,int size);
void print_mathematica_triangle(point2D* A,point2D* B,point2D* C);
void feed_tri_intersection();

#endif
