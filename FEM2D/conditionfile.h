#ifndef CONDFILE_H
#define CONDFILE_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "geometry2D.h"

#define MAX_FUNC_SIZE 1024

#define FAILED 0
#define SUCCESS 1

#define NOCON 0
#define DIRICHLET 1
#define NEUMANN 2
#define ROBIN 3

#define CONST "const"
#define LINEAR "linear"
#define NOISE "noise"
#define DEFAULT "default"
#define FUNCTION "function"
#define SPECIAL "special"
#define IMAGE "image"
#define ATTRIBUTE "attribute"
#define SAW "saw"
#define TRIANG "triang"
#define RECT "rect"
#define RAMP "ramp"
#define LIN_SATURATION "saturation"
#define STEP "step"

typedef struct BOUND_INFO{
	int loops;
	int* Sizes;
	point2D** Coords;
	int** Cond;
	char*** Time_depend;
}bound_info;
 
typedef struct INIT_INFO{
	int size;
	int* Indices;
	char** Cond;
}init_info;

typedef struct BOUND_COND{
	int size;
	int* Cond;
	double* Val;
}bound_cond;

double saw_signal(double x,double period,double offset,double min,double max);
double linear(double x,double period,double min,double max);
double linear_updown_signal(double x,double period,double offset,double min,double max);
double ramp_signal(double x,double period,double offset,double min,double max);
double rect_sgnal(double x,double period,double offset,double min,double max);

int load_condition_file(char* Fullname,bound_info** Bound,int bound_size,init_info* Init);
double parse_expression2D(char* Expression,double x,double y,double t);
double parse_bound_expr(char* Expr,double x,double y,double t);
void free_bound_info(bound_info** Info);
void free_bound_cond(bound_cond** Cond);

#endif
