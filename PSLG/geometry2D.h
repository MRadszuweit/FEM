/* ilb: geometry */

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

point2D* init_point2D(double x,double y);

void vec_mult(point2D* v,double a);
point2D vec_diff(point2D* a,point2D* b);
void vec_add_mult(point2D* a,point2D* b,double fac);
double vec_scalar(point2D* a,point2D* b);
double vec_abs(point2D* v);

void mat_mult(matrix2D* A,double a);

double dist(point2D* a,point2D* b);
void normalize(point2D* v);
point2D get_normal(point2D* P1,point2D* P2,point2D* inner);

#endif
