
#include "geometry2D.h"
 
// Code ///////////////////////////////////////////

point2D* init_point2D(double x,double y){
	point2D* P = malloc(sizeof(point2D));
	P->x = x;
	P->y = y;
	return P;
}

void vec_mult(point2D* v,double a){
	v->x *= a;
	v->y *= a;
}

point2D vec_diff(point2D* a,point2D* b){
	point2D res;
	res.x = a->x-b->x;
	res.y = a->y-b->y;
	return res;
}

void vec_add_mult(point2D* a,point2D* b,double fac){
	a->x += fac*b->x;
	a->y += fac*b->y;
}

void mat_mult(matrix2D* A,double a){
	A->xx *=a;
	A->xy *=a;
	A->yx *=a;
	A->yy *=a;
}

double vec_scalar(point2D* a,point2D* b){
	return (a->x)*(b->x)+(a->y)*(b->y);
}

double vec_abs(point2D* v){
	return sqrt((v->x)*(v->x)+(v->y)*(v->y));
}

double dist(point2D* a,point2D* b){
	return sqrt((a->x-b->x)*(a->x-b->x)+(a->y-b->y)*(a->y-b->y));
}

void normalize(point2D* v){
	double a = vec_abs(v);
	v->x /= a;
	v->y /= a;
}

point2D get_normal(point2D* P1,point2D* P2,point2D* inner){
	point2D res;
	res.x = -(P2->y-P1->y);
	res.y = P2->x-P1->x;
	normalize(&res);
	point2D d = vec_diff(inner,P1);
	if (vec_scalar(&res,&d)>0){vec_mult(&res,-1);}
	return res;
}

