#ifndef SPH2D_H
#define SPH2D_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "geometry2D.h"
#include "linear_algebra.h"

double standard_Gauss2D(point2D* X,point2D* Pos,double h);
void set_SPH2D_function(double (*F)(point2D* X,point2D* Pos,double h),double h);
point2D* regular_move2D(point2D* Reg_mesh,int size_x,int size_y,double dx,double dy,
 double dt,double* vx,double* vy);
void apply_regular_move2D(point2D* Reg_mesh,point2D* Moved_mesh,int size_x,int size_y,
 double dx,double dy,double* Field);
double old_Gauss2D(point2D* X,point2D* Pos,double h);
void set_reference_point(point2D* Ref);
//void init_SPH2D_from_mesh(point2D* Nodes,double* FEM_weights,int size,double R,double h);
//double* SPH2D_move(double* Fields,int size,int flow_index,int field_index,double dt);



#endif
