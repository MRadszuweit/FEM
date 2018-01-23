
#include "geometry2D.h"
 
// Code ///////////////////////////////////////////

point2D init_point2D(double x,double y){
	/*point2D* P = malloc(sizeof(point2D));
	P->x = x;
	P->y = y;
	return P;*/
	point2D p;
	p.x = x;
	p.y = y;
	return p;
}

point2D* clone_point(point2D* P){
	point2D* Res = (point2D*)malloc(sizeof(point2D));
	Res->x = P->x;
	Res->y = P->y;
	return Res;
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

void mat_vec_mult(matrix2D* A,point2D* P){
	double x = P->x;
	double y = P->y;
	P->x = A->xx*x+A->xy*y;
	P->y = A->yx*x+A->yy*y;
}

// B -> A.B
void mat_mat_mult(matrix2D* A,matrix2D* B){
	double xx = A->xx*B->xx+A->xy*B->yx;
	double yx = A->yx*B->xx+A->yy*B->yx;
	double xy = A->xx*B->xy+A->xy*B->yy;
	double yy = A->yx*B->xy+A->yy*B->yy;
	B->xx = xx;
	B->xy = xy;
	B->yx = yx;
	B->yy = yy;
}

void mat_add(matrix2D* A,matrix2D* B,double a){
	A->xx += a*B->xx;
	A->xy += a*B->xy;
	A->yx += a*B->yx;
	A->yy += a*B->yy;
}

matrix2D transpose(matrix2D A){
	matrix2D res;
	res.xx = A.xx;
	res.xy = A.yx;
	res.yx = A.xy;
	res.yy = A.yy;
	return res;
}

matrix2D* tensor_product(point2D* v1,point2D* v2){
	matrix2D* Res = (matrix2D*)malloc(sizeof(matrix2D));
	Res->xx = (v1->x)*(v2->x);
	Res->xy = (v1->x)*(v2->y);
	Res->yx = (v1->y)*(v2->x);
	Res->yy = (v1->y)*(v2->y);
	return Res;
}

matrix2D* unity_matrix2D(){
	matrix2D* Res = (matrix2D*)malloc(sizeof(matrix2D));
	Res->xx = 1.;
	Res->xy = 0.;
	Res->yx = 0.;
	Res->yy = 1.;
	return Res;
}

double determinant_2D(matrix2D* A){
	return A->xx*A->yy-A->xy*A->yx;
}

double trace_2D(matrix2D* A){
	return A->xx+A->yy;
}

// p = p1 + (p2 - p1)*t1 + (p3 - p1)*t2
matrix2D* get_affine_2D(point2D* p1,point2D* p2,point2D* p3){
	matrix2D* Res = (matrix2D*)malloc(sizeof(matrix2D));
	Res->xx = p2->x-p1->x;
	Res->xy = p3->x-p1->x;
	Res->yx = p2->y-p1->y;
	Res->yy = p3->y-p1->y;
	return Res;
}

void get_affine_info_2D(point2D* p1,point2D* p2,point2D* p3,matrix2D* F,matrix2D* IFT,double* detF){
	F->xx = p2->x-p1->x;
	F->xy = p3->x-p1->x;
	F->yx = p2->y-p1->y;
	F->yy = p3->y-p1->y;
	
	*detF = F->xx*F->yy-F->xy*F->yx;
	
	IFT->xx = F->yy/(*detF);
	IFT->xy = -F->yx/(*detF);
	IFT->yx = -F->xy/(*detF);
	IFT->yy = F->xx/(*detF);
}

matrix2D* get_transpose_2D(matrix2D* A){
	matrix2D* Res = (matrix2D*)malloc(sizeof(matrix2D));
	memcpy(Res,A,sizeof(matrix2D));
	Res->xy = A->yx;
	Res->yx = A->xy;
	return Res;
}

matrix2D* get_inverse_transposed_2D(matrix2D* A){
	double detA = determinant_2D(A);
	if (detA==0) return NULL;
	
	matrix2D* Res = (matrix2D*)malloc(sizeof(matrix2D));
	Res->xx = A->yy/detA;
	Res->xy = -A->yx/detA;
	Res->yx = -A->xy/detA;
	Res->yy = A->xx/detA;
	return Res;
}

double contraction(matrix2D* A,matrix2D* B){
	return (A->xx)*(B->xx)+(A->xy)*(B->yx)+(A->yx)*(B->xy)+(A->yy)*(B->yy);
}

double vec_scalar(point2D* a,point2D* b){
	return (a->x)*(b->x)+(a->y)*(b->y);
}

double vec_abs(point2D* v){
	return sqrt((v->x)*(v->x)+(v->y)*(v->y));
}

double Vec_abs(point2D v){
	return sqrt(v.x*v.x+v.y*v.y);
}

double dist(point2D* a,point2D* b){
	return sqrt((a->x-b->x)*(a->x-b->x)+(a->y-b->y)*(a->y-b->y));
}

double sqrdist(point2D* a,point2D* b){
	return (a->x-b->x)*(a->x-b->x)+(a->y-b->y)*(a->y-b->y);
}

double vec_cross2D(point2D* a,point2D* b){
	return a->y*b->x-a->x*b->y;
}

void normalize(point2D* v){
	double a = vec_abs(v);
	v->x /= a;
	v->y /= a;
}

point2D* get_normalized(point2D* v){
	double a = vec_abs(v);
	point2D* Res = clone_point(v);
	vec_mult(Res,(double)1/a);
	return Res;
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

point2D Get_normal(point2D p1,point2D p2,point2D inner){
	point2D res;
	res.x = -(p2.y-p1.y);
	res.y = p2.x-p1.x;
	normalize(&res);
	point2D d = vec_diff(&inner,&p1);
	if (vec_scalar(&res,&d)>0){vec_mult(&res,-1);}
	return res;
}

point2D ortho_projection(point2D projector,point2D v){
	point2D res;
	point2D n = init_point2D(projector.x,projector.y);
	normalize(&n);
	res.x = (1-n.x*n.x)*v.x-n.x*n.y*v.y;
	res.y = -n.y*n.x*v.x+(1-n.y*n.y)*v.y;
	return res;
}

int in_triangle(point2D* A,point2D* B,point2D* C,point2D* P){  // Achtung: -1 by false, nicht 0 ! 
	point2D a,b,c,p1,p2;
	a = vec_diff(B,A);
	b = vec_diff(C,B);
	c = vec_diff(A,C);
	p1 = ortho_projection(a,vec_diff(P,A));
	p2 = ortho_projection(a,vec_diff(C,A));
	if (vec_scalar(&p1,&p2)<0){return -1;}
	p1 = ortho_projection(b,vec_diff(P,B));
	p2 = ortho_projection(b,vec_diff(A,B));
	if (vec_scalar(&p1,&p2)<0){return -1;}
	p1 = ortho_projection(c,vec_diff(P,C));
	p2 = ortho_projection(c,vec_diff(B,C));
	if (vec_scalar(&p1,&p2)<0){return -1;}
	return 1;
}

int in_triangle_2D(point2D* A,point2D* B,point2D* C,point2D* P,double tol){  // Achtung: 0 by false ! 
	point2D a,b,c,p1,p2;
	double tol2 = tol*tol;
	if (sqrdist(A,P)<tol2 || sqrdist(B,P)<tol2 || sqrdist(B,P)<tol2) return 1;
	
	a = vec_diff(B,A);
	b = vec_diff(C,B);
	c = vec_diff(A,C);
	p1 = ortho_projection(a,vec_diff(P,A));
	p2 = ortho_projection(a,vec_diff(C,A));
	if (vec_scalar(&p1,&p2)<-tol){return 0;}
	p1 = ortho_projection(b,vec_diff(P,B));
	p2 = ortho_projection(b,vec_diff(A,B));
	if (vec_scalar(&p1,&p2)<-tol){return 0;}
	p1 = ortho_projection(c,vec_diff(P,C));
	p2 = ortho_projection(c,vec_diff(B,C));
	if (vec_scalar(&p1,&p2)<-tol){return 0;}
	return 1;
}

point2D* line_intersection_2D(point2D* P1,point2D* P2,point2D* Q1,point2D* Q2,double tol){
	double tol2 = tol*tol;
	if (sqrdist(P1,Q1)<tol2 || sqrdist(P1,Q2)<tol2 || sqrdist(P2,Q1)<tol2 || sqrdist(P2,Q2)<tol2) return NULL;
	double det = -(P1->y-P2->y)*(Q1->x-Q2->x)+(P1->x-P2->x)*(Q1->y-Q2->y);
	if (fabs(det)>tol2){
		double t = (-Q1->y*Q2->x+P1->y*(-Q1->x+Q2->x)+P1->x*(Q1->y-Q2->y)+Q1->x*Q2->y)/det;
		double s = -(-P2->y*Q1->x+P1->y*(-P2->x+Q1->x)+P1->x*(P2->y-Q1->y)+P2->x*Q1->y)/det;
	
		if (t>tol && t<1.-tol && s>tol && s<1.-tol){		
			point2D d = vec_diff(P2,P1);
			vec_mult(&d,t);
			vec_add_mult(&d,P1,1.);
			return clone_point(&d);
		}
		else return NULL;
	}
	else return NULL;
}

point2D* line_intersection_positive_2D(point2D* P1,point2D* P2,point2D* Q1,point2D* Q2,double tol){
	
	double det = -(P1->y-P2->y)*(Q1->x-Q2->x)+(P1->x-P2->x)*(Q1->y-Q2->y);
	if (fabs(det)>tol*tol){
		double t = (-Q1->y*Q2->x+P1->y*(-Q1->x+Q2->x)+P1->x*(Q1->y-Q2->y)+Q1->x*Q2->y)/det;
		double s = -(-P2->y*Q1->x+P1->y*(-P2->x+Q1->x)+P1->x*(P2->y-Q1->y)+P2->x*Q1->y)/det;
	
		if (t>-tol && t<1.+tol && s>-tol && s<1.+tol){		
			point2D d = vec_diff(P2,P1);
			vec_mult(&d,t);
			vec_add_mult(&d,P1,1.);
			return clone_point(&d);
		}
		else return NULL;
	}
	else return NULL;
}

double line_ray_intersection_2D(point2D* P1,point2D* P2,point2D* Q,point2D* N,double tol){
	double tol2 = tol*tol;																	// result t is the  barycentric coordinate
	if (sqrdist(P1,Q)<tol2) return 0;														// between P1 and P2, such that
	if (sqrdist(P2,Q)<tol2) return 1.;														// P(t=0) = P1
	double det = N->y*(P2->x-P1->x)+N->x*(P1->y-P2->y);									
	if (fabs(det)>tol2){
		double t = (N->y*(Q->x-P1->x)+N->x*(P1->y-Q->y))/det;
		if (fabs(t)<tol) t = 0.0;
		if (fabs(t-1)<tol) t = 1.0;
		return t;
	}
	else{
		det = Q->y*(P2->x-P1->x)+Q->x*(P1->y-P2->y)+P1->x*P2->y-P1->y*P2->x;
		if (fabs(det)>tol2) return DBL_MAX; else return 0;
	}
}

double* barycentric_iso_line(double a, double b, double c,point2D* Grad_ab,double tol){		// returns new (a,b,c)
	point2D A = {.x=1.0,.y=0};														// x: a coordinate, y: b coordinate
	point2D B = {.x=0,.y=1.0};		
	point2D C = {.x=0,.y=0};
	
	double t;
	
	double* Res = (double*)malloc(3*sizeof(double));
	point2D Q = {.x=a,.y=b};
	point2D N = {.x=-Grad_ab->y,.y=Grad_ab->x};
	if (vec_abs(&N)==0) goto NO_INTERSECTION;
	normalize(&N);
	
	if (fabs(a)<tol){		
		if (fabs(b)<tol){		
			t = line_ray_intersection_2D(&A,&B,&C,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = 1.-t;
				Res[1] = t;
				Res[2] = 0;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else if (fabs(c)<tol){
			t = line_ray_intersection_2D(&A,&C,&B,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = 1.-t;
				Res[1] = 0;
				Res[2] = t;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else{
			t = line_ray_intersection_2D(&A,&B,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = 1.-t;
				Res[1] = t;
				Res[2] = 0;
				return Res;
			}
			t = line_ray_intersection_2D(&A,&C,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = 1.-t;
				Res[1] = 0;
				Res[2] = t;
				return Res;
			}			
			else goto NO_INTERSECTION;
		}
	}
	else if (fabs(b)<tol){
		if (fabs(a)<tol){
			t = line_ray_intersection_2D(&B,&A,&C,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = t;
				Res[1] = 1.-t;
				Res[2] = 0;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else if (fabs(c)<tol){
			t = line_ray_intersection_2D(&B,&C,&A,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = 0;
				Res[1] = 1.-t;
				Res[2] = t;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else{
			t = line_ray_intersection_2D(&B,&A,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = t;
				Res[1] = 1.-t;
				Res[2] = 0;
				return Res;
			}
			t = line_ray_intersection_2D(&B,&C,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = 0;
				Res[1] = 1.-t;
				Res[2] = t;
				return Res;
			}			
			else goto NO_INTERSECTION;
		}
	}
	else if (fabs(c)<tol){
		if (fabs(a)<tol){
			t = line_ray_intersection_2D(&C,&A,&B,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = t;
				Res[1] = 0;
				Res[2] = 1.-t;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else if (fabs(b)<tol){
			t = line_ray_intersection_2D(&C,&B,&A,&N,tol);
			if (t>=0 && t<=1.0){
				Res[0] = 0;
				Res[1] = t;
				Res[2] = 1.-t;
				return Res;
			}
			else goto NO_INTERSECTION;
		}
		else{
			t = line_ray_intersection_2D(&C,&A,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = t;
				Res[1] = 0;
				Res[2] = 1.-t;
				return Res;
			}
			t = line_ray_intersection_2D(&C,&B,&Q,&N,tol);
			if (t>=0 && t<1.0){
				Res[0] = 0;
				Res[1] = t;
				Res[2] = 1.-t;
				return Res;
			}			
			else goto NO_INTERSECTION;
		}
	}
	else printf("Warning: initial point not a simplex point (%e,%e,%e)\n",a,b,c);
	
NO_INTERSECTION:
		
	free(Res);
	return NULL;
}

int count_nonzero(point2D** List,int len){
	int i;
	int res = 0;
	for (i=0;i<len;i++) if (List[i]!=NULL) res++;
	return res;
}

int chance_to_overlap(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2){
	double min1x = A1->x;
	double max1x = min1x;
	if (B1->x<min1x) min1x = B1->x;
	if (B1->x>max1x) max1x = B1->x;
	if (C1->x<min1x) min1x = C1->x;
	if (C1->x>max1x) max1x = C1->x;
	
	double min2x = A2->x;
	double max2x = min2x;
	if (B2->x<min2x) min2x = B2->x;
	if (B2->x>max2x) max2x = B2->x;
	if (C2->x<min2x) min2x = C2->x;
	if (C2->x>max2x) max2x = C2->x;
	
	if (max1x<=min2x || min1x>=max2x) return 0;
	
	double min1y = A1->y;
	double max1y = min1y;
	if (B1->y<min1y) min1y = B1->y;
	if (B1->y>max1y) max1y = B1->y;
	if (C1->y<min1y) min1y = C1->y;
	if (C1->y>max1y) max1y = C1->y;
	
	double min2y = A2->y;
	double max2y = min2y;
	if (B2->y<min2y) min2y = B2->y;
	if (B2->y>max2y) max2y = B2->y;
	if (C2->y<min2y) min2y = C2->y;
	if (C2->y>max2y) max2y = C2->y;
	
	if (max1y<=min2y || min1y>=max2y) return 0;
	
	return 1;
}

point2D*** get_triangle_intersection_matrix(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,int* int_num,double tol){
	int i,j;
	point2D*** Res = (point2D***)malloc(3*sizeof(point2D**));
	for (i=0;i<3;i++) Res[i] = (point2D**)malloc(3*sizeof(point2D*));
	
	Res[0][0] = line_intersection_2D(A1,B1,A2,B2,tol);
	Res[0][1] = line_intersection_2D(A1,B1,B2,C2,tol);
	Res[0][2] = line_intersection_2D(A1,B1,C2,A2,tol);
	
	
	Res[1][0] = line_intersection_2D(B1,C1,A2,B2,tol);
	Res[1][1] = line_intersection_2D(B1,C1,B2,C2,tol);
	Res[1][2] = line_intersection_2D(B1,C1,C2,A2,tol);
	
	Res[2][0] = line_intersection_2D(C1,A1,A2,B2,tol);
	Res[2][1] = line_intersection_2D(C1,A1,B2,C2,tol);
	Res[2][2] = line_intersection_2D(C1,A1,C2,A2,tol);
	
	*int_num = 0;
	for (i=0;i<3;i++) (*int_num) += count_nonzero(Res[i],3);
	
	return Res;
}

point2D** inside_triangle_points(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,int* ins_num,int* all_in,double tol){
	
	int i1,i2;
	point2D** Res = (point2D**)malloc(6*sizeof(point2D*));
	*ins_num = 0;
	
	i1 = 0;
	if (in_triangle_2D(A2,B2,C2,A1,tol)){
		(*ins_num)++;
		i1++;
		Res[(*ins_num)-1] = A1;
	}
	if (in_triangle_2D(A2,B2,C2,B1,tol)){
		(*ins_num)++;
		i1++;
		Res[(*ins_num)-1] = B1;
	}
	if (in_triangle_2D(A2,B2,C2,C1,tol)){
		(*ins_num)++;
		i1++;
		Res[(*ins_num)-1] = C1;
	}
	
	i2 = 0;
	if (in_triangle_2D(A1,B1,C1,A2,tol)){
		(*ins_num)++;
		i2++;
		Res[(*ins_num)-1] = A2;
	}
	if (in_triangle_2D(A1,B1,C1,B2,tol)){
		(*ins_num)++;
		i2++;
		Res[(*ins_num)-1] = B2;
	}
	if (in_triangle_2D(A1,B1,C1,C2,tol)){
		(*ins_num)++;
		i2++;
		Res[(*ins_num)-1] = C2;
	}

	if (i1==3 || i2==3) *all_in =1; else *all_in = 0;
	return Res;
}

point2D* get_ith_nonzero(point2D*** M,int i){
	int j,k,l;
	point2D* Res = NULL;
	l = 0;
	for (j=0;j<3;j++){
		for (k=0;k<3;k++) if (M[j][k]!=NULL){
			l++;
			if (l==i){
				Res = M[j][k];
				goto FINALIZE;
			}
		}
	}
	FINALIZE:
	return Res;
}

void get_affine_trafo_2D(point2D* P1,point2D* P2,point2D* P3,double** F,double** G){ 		// x->F.x+g, (0,0) -> P3
	point2D* Col1 = clone_point(P1);
	point2D* Col2 = clone_point(P2);
	vec_add_mult(Col1,P3,-1.);
	vec_add_mult(Col2,P3,-1.);
	
	*F = (double*)malloc(4*sizeof(double));
	(*F)[0] = Col1->x;
	(*F)[2] = Col1->y;
	(*F)[1] = Col2->x;
	(*F)[3] = Col2->y;
	
	*G = (double*)malloc(2*sizeof(double));
	(*G)[0] = P3->x;
	(*G)[1] = P3->y;
	
	free(Col1);
	free(Col2);
}

int congruent_triangles_2D(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,double tol){
	
	double d1 = sqrdist(A1,A2);
	double d2 = sqrdist(A1,B2);
	double d3 = sqrdist(A1,C2);
	
	if (d1<tol || d2<tol || d3<tol){
		if (d1<tol){
			d2 = sqrdist(B1,B2);
			d3 = sqrdist(B1,C2);
			if (d2<tol && sqrdist(C1,C2)<tol) return 1;
			if (d3<tol && sqrdist(C1,B2)<tol) return 1;
		}
		if (d2<tol){
			d2 = sqrdist(B1,A2);
			d3 = sqrdist(B1,C2);
			if (d2<tol && sqrdist(C1,C2)<tol) return 1;
			if (d3<tol && sqrdist(C1,A2)<tol) return 1;
		}
		if (d3<tol){
			d2 = sqrdist(B1,A2);
			d3 = sqrdist(B1,B2);
			if (d2<tol && sqrdist(C1,B2)<tol) return 1;
			if (d3<tol && sqrdist(C1,A2)<tol) return 1;
		}
		return 0;
	}
	else return 0;
}

int congruent_triangle(int i1,int j1,int k1,int i2, int j2,int k2){
	if (i1==i2){
		if (j1==j2){
			if (k1==k2) return 1;else return 0;
		}
		else if (j1==k2){
			if (j2==k1) return 1; else return 0;
		}		
		else return 0;
	}
	else if (i1==j2){
		if (j1==i2){
			if (k1==k2) return 1;else return 0;
		}
		else if (j1==k2){
			if (k1==i2) return 1;else return 0;		
		}
		else return 0;
	}
	else if (i1==k2){
		if (k1==i2){
			if (j1==j2) return 1;else return 0;
		}
		else if (k1==j2){
			if (j1==i2) return 1;else return 0;		
		}
		else return 0;
	}
	else return 0;
}

int test_intersections(point2D** Pool,int size,int index,point2D* Start,point2D* End,double tol){
	int i,j;
	point2D* S;
	for (i=0;i<size;i++) if (i!=index){
		for (j=0;j<size;j++) if (i!=j && j!=index){
			S = line_intersection_2D(Pool[index],Start,Pool[i],Pool[j],tol);
			if (S!=NULL){
				free(S);
				return 1;
			}
		}
		S = line_intersection_2D(Pool[index],Start,Pool[i],End,tol);
		if (S!=NULL){
			free(S);
			return 1;
		}
	}
	return 0;
}

void delete_duplicate_points(point2D** List,int* size,double tol){
	int i,j;
	double d;
	int* Equal = (int*)malloc((*size)*sizeof(int));
	point2D* Temp = (point2D*)malloc((*size)*sizeof(point2D));
	int sum = 0;
	for (i=0;i<*size;i++){
		Equal[i] = 0;
		Temp[i] = *(List[i]);
		for (j=0;j<i;j++){
			d = sqrdist(List[i],List[j]);
			if (d<tol*tol){
				Equal[i] = 1;
				sum++;
			}
		}
	}
	if (sum!=0){
		j = 0;
		for (i=0;i<*size;i++){
			if (!Equal[i]){
				(*List[j]) = Temp[i];
				j++;
			}
		}
		*size = j;
	}
	
	free(Equal);
	free(Temp);
}

double mod2Pi(double x){
	double d = x/(2.*M_PI);
	return x-2.*M_PI*floor(d);	
}

void order_convex_polygon(point2D** Verts,int size,double tol){
	int i,j,k,intersection;
	double a,a0,min,last;
	
	point2D center = init_point2D(0,0);
	for (i=0;i<size;i++) vec_add_mult(&center,Verts[i],1.);
	vec_mult(&center,(double)1/size);
	
	point2D* New_list = (point2D*)malloc(size*sizeof(point2D));
	New_list[0] = init_point2D(Verts[0]->x,Verts[0]->y);
	a0 = atan2(Verts[0]->y-center.y,Verts[0]->x-center.x);
	last = 0;
	for (i=1;i<size;i++){
		min = 2.*M_PI;
		for (j=1;j<size;j++){
			a = mod2Pi(atan2(Verts[j]->y-center.y,Verts[j]->x-center.x)-a0);		
			if (a<min && a>last){
				min = a;
				k = j;
			}
		}
		New_list[i] =  init_point2D(Verts[k]->x,Verts[k]->y);
		last = min;
	}
	
	for (i=0;i<size;i++){
		Verts[i]->x = New_list[i].x;
		Verts[i]->y = New_list[i].y;
	}
	free(New_list);
}

void print_mathematica_polygon(point2D** Poly,int size){
	int i;
	FILE* file = fopen("/Home/damage/radszuwe/Daten/polygon","w");
	fprintf(file,"Graphics[{,Polygon[{");
	for (i=0;i<size-1;i++) fprintf(file,"{%f,%f},",Poly[i]->x,Poly[i]->y);
	fprintf(file,"{%f,%f}",Poly[size-1]->x,Poly[size-1]->y);
	fprintf(file,"}]}]\n");
	fclose(file);
}

void print_mathematica_triangle(point2D* A,point2D* B,point2D* C){
	int i;
	FILE* file = fopen("/Home/damage/radszuwe/Daten/polygon","w");
	fprintf(file,"Graphics[{,Polygon[{");
	fprintf(file,"{%f,%f},",A->x,A->y);
	fprintf(file,"{%f,%f},",B->x,B->y);
	fprintf(file,"{%f,%f}",C->x,C->y);
	fprintf(file,"}]}]\n");
	fclose(file);
}

void triangle_intersection_2D(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,
 point2D*** Poly_list,int* poly_size,double tol){

	int i,j,old_size,intersections,inside_number;
	
	point2D*** M = get_triangle_intersection_matrix(A1,B1,C1,A2,B2,C2,&intersections,tol);
	
	int all_in = 0;
	point2D** I = inside_triangle_points(A1,B1,C1,A2,B2,C2,&inside_number,&all_in,tol);
	
	*poly_size = inside_number+intersections;	
	
	if (*poly_size!=0) *Poly_list = (point2D**)realloc(*Poly_list,(*poly_size)*sizeof(point2D*)); else *Poly_list = NULL;
	for (i=0;i<inside_number;i++) (*Poly_list)[i] = clone_point(I[i]);
	for (i=inside_number;i<(*poly_size);i++) (*Poly_list)[i] = clone_point(get_ith_nonzero(M,i-inside_number+1));
		
	old_size = *poly_size;
	delete_duplicate_points(*Poly_list,poly_size,tol);
	
	if (*poly_size<3){
		for (i=0;i<old_size;i++) free((*Poly_list)[i]);
		free(*Poly_list);
		*Poly_list = NULL;
		*poly_size = 0;
	}
	else if (*poly_size<old_size){
		for (i=*poly_size;i<old_size;i++) free((*Poly_list)[i]);
		*Poly_list = (point2D**)realloc(*Poly_list,(*poly_size)*sizeof(point2D*));
	}
	
	if (*poly_size>3) order_convex_polygon(*Poly_list,*poly_size,tol);
	/*if (*poly_size>6){
		printf("Warning: Higher polygonal of %d edges encountered\n",*poly_size);
	}*/
	
	//clean
	for (i=0;i<3;i++){
		for (j=0;j<3;j++) free(M[i][j]);
		free(M[i]);
	}
	free(M);
	free(I);
}

void feed_tri_intersection(){
	point2D a1 = init_point2D(0,0);
	point2D b1 = init_point2D(4,0);
	point2D c1 = init_point2D(2,2);
	
	point2D a2 = init_point2D(1,1);
	point2D b2 = init_point2D(3,1);
	point2D c2 = init_point2D(2,0);
	
	int poly_size = 0;
	point2D** Poly = NULL;
	
	triangle_intersection_2D(&a1,&b1,&c1,&a2,&b2,&c2,&Poly,&poly_size,1e-8);
	
	exit(0);
}

double polygon_area(point2D* Edges,int size){
	int i;
	double sum = 0;
	int j = 1;
	for (i=0;i<size;i++){
		sum += (Edges[i].x-Edges[j].x)*(Edges[i].y+Edges[j].y);
		if (i<size-2) j++; else j = 0;
	}
	return fabs(sum/2);
}

double Polygon_area(point2D** Poly,int size){
	int i;
	point2D* E = (point2D*)malloc(size*sizeof(point2D));
	for (i=0;i<size;i++) E[i] = *(Poly[i]);
	double res = polygon_area(E,size);
	free(E);
	return res;
}

double Polygon_Area(point2D* Points,int* Poly,int size){
	int i;
	point2D* E = (point2D*)malloc(size*sizeof(point2D));
	for (i=0;i<size;i++) E[i] = init_point2D(Points[Poly[i]].x,Points[Poly[i]].y);
	double res = polygon_area(E,size);
	free(E);
	return res;
}

void point2D_sort(int* List,point2D** Values,int size,point2D* Ref){  // divide & conquer
	if (size>1){									  // List wird permutiert
		int i;										  // Values bleibt unverändert
		int s_size = 0;
		int b_size = 0;
		int a = List[0];
		int* Smaller = (int*)malloc(s_size*sizeof(int));
		int* Bigger = (int*)malloc(b_size*sizeof(int));
		for (i=1;i<size;i++) if (dist(Values[List[i]],Ref)>dist(Values[a],Ref)){
			b_size++;
			Bigger = (int*)realloc(Bigger,b_size*sizeof(int));
			Bigger[b_size-1] = List[i];
		}
		else{
			s_size++;
			Smaller = (int*)realloc(Smaller,s_size*sizeof(int));
			Smaller[s_size-1] = List[i];
		}
		if (1+b_size+s_size != size) printf("Fehler bei double_sort\n");
		point2D_sort(Smaller,Values,s_size,Ref);
		point2D_sort(Bigger,Values,b_size,Ref);
		for (i=0;i<s_size;i++) List[i] = Smaller[i];
		List[s_size] = a;
		for (i=0;i<b_size;i++) List[s_size+i+1] = Bigger[i];
		free(Smaller);
		free(Bigger);
	}
}

double Area_form2D(point2D* A,point2D* B,point2D* C){
	return ((A->x-C->x)*(B->y-C->y)-(B->x-C->x)*(A->y-C->y))/2.;
}

void print_point_list2D(point2D* V,int n,char* Name){
	int i;
	if (V!=NULL){
		char Dir[512];
		sprintf(Dir,"/Home/damage/radszuwe/Daten/%s",Name);
		FILE* file = fopen(Dir,"w");
		for (i=0;i<n;i++){
			fprintf(file,"%f\t%f\n",V[i].x,V[i].y);
		}
		fclose(file);
	}
}

double create_arc(int n,double r,double angle,point2D* data){
	int i;
	double phi;
	for (i=0;i<n;i++){
		phi = (double)angle*i/n;
		data[i].x = r*sin(phi);
		data[i].y = r*cos(phi);
	}
	double d = dist(&data[0],&data[1]);
	printf("area magnitude: %f\n",d*d/2);
	return d*d/2;
}

void create_poly_file(char* name,point2D* data,int total_size){   //,int* attr_sizes,point2D* attr_points,int attr_num){
	int i;
	point2D p;
	FILE* File = fopen(name,"w");
	//int n = 0;
	//for (k=0;k<attr_num;k++) n += attr_sizes[k]; 
	fprintf(File,"%d %d %d %d\n",total_size,2,0,0);
	for (i=0;i<total_size;i++){
		p = data[i];
		fprintf(File,"%d %f %f\n",i,p.x,p.y);
	}
	fprintf(File,"%d %d\n",total_size,0);

	for (i=0;i<total_size-1;i++){
		fprintf(File,"%d %d %d\n",i,i,i+1);
	}
	fprintf(File,"%d %d %d\n",total_size,total_size-1,0);
	fprintf(File,"%d\n",0);
	/*int start = 0;
	int k;
	for (k=0;k<attr_num;k++){
		fprintf(File,"%d %d %d\n",start,start+attr_sizes[k]-1,start);
		for (i=start+1;i<start+attr_sizes[k];i++){
			fprintf(File,"%d %d %d\n",i,i-1,i);
		}
		start += attr_sizes[k];
	}
	fprintf(File,"%d\n",0);
	if (attr_num>1){
		fprintf(File,"%d\n",attr_num);
		for (i=0;i<attr_num;i++){
			fprintf(File,"%d %f %f %d %d\n",i,attr_points[i].x,attr_points[i].y,i,-1);
		}
	}*/
	fclose(File);
}

void create_nonconvex_poly_file(char* name,point2D** data,int* Sizes,int number,point2D* Outer){
	int i,j,k;
	point2D p;
	
	int total_size = 0;
	for (i=0;i<number;i++) total_size += Sizes[i]; 
	FILE* File = fopen(name,"w");
	fprintf(File,"%d %d %d %d\n",total_size,2,0,0);
	
	k = 0;
	for (j=0;j<number;j++){
		for (i=0;i<Sizes[j];i++){
			p = data[j][i];
			fprintf(File,"%d %f %f\n",k,p.x,p.y);
			k++;
		}
	}
	fprintf(File,"%d %d\n",total_size,0);
	
	k = 0;
	for (j=0;j<number;j++){
		for (i=0;i<Sizes[j]-1;i++){
			fprintf(File,"%d %d %d\n",k,k,k+1);
			k++;
		}
		fprintf(File,"%d %d %d\n",total_size,k,k-Sizes[j]+1);
		k++;
	}
	
	fprintf(File,"%d\n",number-1);
	for (i=0;i<number-1;i++) fprintf(File,"%d %f %f\n",i,Outer[i].x,Outer[i].y);
	
	fclose(File);
}

void create_condition_template(char* name,point2D** data,int number,int* Var_indices,int var_num){
	FILE* File = fopen(name,"w");
	if (File!=NULL){
		int i,j;
		fprintf(File,"begin(BC)\n");
		for (i=0;i<var_num;i++){			
			fprintf(File,"var %d\n",Var_indices[i]);
			for (j=0;j<number;j++){
				fprintf(File,"(%.6f,%.6f)\t NEUMANN(const,0)\n",data[j][0].x,data[j][0].y);
			}
		}
		fprintf(File,"end(BC)\n\n");
		
		fprintf(File,"begin(IC)\n");
		fprintf(File,"end(IC)\n");
	}	
}

void create_attribute_poly_file(char* name,point2D** data,int* Sizes,int number,point2D* Inner,double* Attr){
	int i,j,k;
	point2D p;
	
	int total_size = 0;
	for (i=0;i<number;i++) total_size += Sizes[i]; 
	FILE* File = fopen(name,"w");
	fprintf(File,"%d %d %d %d\n",total_size,2,1,0);
	
	k = 0;
	for (j=0;j<number;j++){
		for (i=0;i<Sizes[j];i++){
			p = data[j][i];
			fprintf(File,"%d %f %f %f\n",k,p.x,p.y,Attr[j]);
			k++;
		}
	}
	fprintf(File,"%d %d\n",total_size,0);
	
	k = 0;
	for (j=0;j<number;j++){
		for (i=0;i<Sizes[j]-1;i++){
			fprintf(File,"%d %d %d\n",k,k,k+1);
			k++;
		}
		fprintf(File,"%d %d %d\n",total_size,k,k-Sizes[j]+1);
		k++;
	}
	fprintf(File,"%d\n",0);
	fprintf(File,"%d\n",number);
	for (i=0;i<number;i++) fprintf(File,"%d %f %f %f\n",i,Inner[i].x,Inner[i].y,Attr[i]);
	
	fclose(File);
}

void read_segments_from_polyfile(FILE* file,int* Indices,int size){
	int i,ind,ind1,ind2;
	size_t bsize;
	char* Buffer = NULL;
	
	for (i=0;i<size;i++){
		getline(&Buffer,&bsize,file);
		ind = atoi(strtok(Buffer," "));
		ind1 = atoi(strtok(NULL," \n"));
		ind2 = atoi(strtok(NULL," \n"));
		Indices[ind] = ind1;
		Indices[ind+size] = ind2;
		free(Buffer);
		Buffer = NULL;
	}
} 

int segments_connected(int* Segments,int size,int i,int j){
	
	int i1 = Segments[i];
	int i2 = Segments[i+size];
	int j1 = Segments[j];
	int j2 = Segments[j+size];
	
	if (i1==j1 || i2==j1){
		return j2;
	}
	else if (i1==j2 || i2==j2){
		return j1;
	}
	else return -1;
}

int get_common_point2D(int* Segments,int size,int i,int j){
	int res = segments_connected(Segments,size,i,j);
	if (res<0) return res;
	res = (res==Segments[j]) ? res = Segments[j+size] : Segments[j];
	return res;
}

void get_all_connected_segments(int* Segments,int size,int current,int** List,int* len){
	int i;
	
	*len = 0;
	if (List!=NULL) *List = NULL;
	for (i=0;i<size;i++) if (i!=current && segments_connected(Segments,size,current,i)>=0){
		(*len)++;
		if (List!=NULL){
			*List = (int*)realloc(*List,(*len)*sizeof(int));			
			(*List)[(*len)-1] = i;		
		}
	}
}

static int segment_in_list(int* List,int size,int index){
	int i;
	for (i=0;i<size;i++) if (index==List[i]) return 1;
	return 0;
}

static void list_or(int* A,int *B,int size){										// A-> A||B
	int i;
	for (i=0;i<size;i++) A[i] = (A[i] || B[i]);
}

int choose_segment(point2D* Points,int* Segments,int size,int* Prev,int prev_size,int* List,int len,int current,int orientation){
	int i,common,next,prev;
	double a;
	point2D* V1;
	point2D* V2;
	
	if (len==1) return List[0];
	
	int k = Segments[current];
	int l = Segments[current+size];
	int res = -1;
	double a_extr = (orientation>0) ? -1 : 1;
	for (i=0;i<len;i++) if (!segment_in_list(Prev,prev_size,List[i])){		
		if (k==Segments[List[i]] || k==Segments[List[i]+size]){
			common = k;
			prev = l;
			next = (k!=Segments[List[i]]) ? Segments[List[i]] : Segments[List[i]+size];
		}
		else{
			common = l;
			prev = k;
			next = (l!=Segments[List[i]]) ? Segments[List[i]] : Segments[List[i]+size];			
		}
		V1 = clone_point(&(Points[common]));
		vec_add_mult(V1,&(Points[prev]),-1.);
		normalize(V1);
		V2 = clone_point(&(Points[next]));
		vec_add_mult(V2,&(Points[common]),-1.);
		normalize(V2);
		a = vec_cross2D(V1,V2);
		if (i==0){
			res = List[i];
			a_extr = a;
		}
		else{
			if (orientation>0 && a>=a_extr){
				res = List[i];
				a_extr = a;
			}
			if (orientation<=0 && a<=a_extr){
				res = List[i];
				a_extr = a;
			}
		}
		free(V1);
		free(V2);
	}
	return res;
}

int* polygon_from_segments(point2D* Points,int* Segments,int size,int start,int orientation,int** Poly,int* poly_size){
	int i,j,k,len,next,o;
	int* List;
	
	int* Prev = NULL;
	int prev_size = 0;
	int* Res = (int*)malloc(size*sizeof(int));
	
	*poly_size = 0;
	(*Poly) = (int*)malloc(size*sizeof(int));
	for (i=0;i<size;i++) Res[i] = 0;
	
	i = start;
	Res[start] = 1;
	do{
		get_all_connected_segments(Segments,size,i,&List,&len);
		if (len>0){
			o = (i==start) ? 1 : orientation;
			next = choose_segment(Points,Segments,size,Prev,prev_size,List,len,i,o);
			if (next<0){
				printf("strange segment %d:\n",i);
				printf("len: %d, polysize: %d\n, List[0]=%d, List[1]=%d",len,*poly_size,List[0],List[1]);
				exit(0);
			}
			(*Poly)[*poly_size] = get_common_point2D(Segments,size,i,next);
			(*poly_size)++;	
			//printf("segment: %d, %d\n",Segments[next],Segments[next+size]);
			Res[next] = 1;
			Prev = (int*)realloc(Prev,len*sizeof(int));
			Prev[0] = i;
			k = 1;
			for (j=0;j<len;j++) if (List[j]!=next){
				Prev[k] = List[j];
				k++;
			}				
			prev_size = k;
		}
		else{			
				printf("PSLG segments not closed at segment %d-> abort\n",i);
				exit(0);			
		}
		i = next;
		free(List);
	}while(next!=start && (*poly_size)<size);
	
	*Poly = (int*)realloc(*Poly,(*poly_size)*sizeof(int));
	if (Prev!=NULL) free(Prev);
	return Res;
}

void Polygons_from_segments(point2D* Points,int* Segments,int size,int*** Polys,int** Poly_sizes,int* poly_num,int quiet){
	int i,j,len;
	int* Poly_pos;
	int* Poly_neg;
	int* Set_pos;
	int* Set_neg;
	
	int* Set = (int*)malloc(size*sizeof(int));
	for (i=0;i<size;i++) Set[i] = 0;
	int seed = 0;
	
	*poly_num = 0;
	*Polys = NULL;
	*Poly_sizes = NULL;
	
	if (!quiet) printf("search for domains:\n");
	while(seed>=0){
		int len_pos = 0;
		int len_neg = 0;
		Poly_pos = NULL;
		Poly_neg = NULL;
		Set_neg = polygon_from_segments(Points,Segments,size,seed,-1,&Poly_neg,&len_neg);
		Set_pos = polygon_from_segments(Points,Segments,size,seed,1,&Poly_pos,&len_pos);
		
		double a_neg = fabs(Polygon_Area(Points,Poly_neg,len_neg));
		double a_pos = fabs(Polygon_Area(Points,Poly_pos,len_pos));
		if (len_neg>0 || len_pos>0){
			(*poly_num)++;
			(*Polys) = (int**)realloc(*Polys,(*poly_num)*sizeof(int*));
			(*Poly_sizes) = (int*)realloc(*Poly_sizes,(*poly_num)*sizeof(int));
			if (a_neg<a_pos){
				(*Polys)[(*poly_num)-1] = Poly_neg;
				(*Poly_sizes)[(*poly_num)-1] = len_neg;
				list_or(Set,Set_neg,size);
				if (!quiet) printf("polygon #%d: area=%f\n",(*poly_num)-1,a_neg);
				free(Poly_pos);
			}
			else{
				(*Polys)[(*poly_num)-1] = Poly_pos;
				(*Poly_sizes)[(*poly_num)-1] = len_pos;
				list_or(Set,Set_pos,size);
				if (!quiet) printf("polygon #%d: area=%f\n",(*poly_num)-1,a_pos);
				free(Poly_neg);
			}			
			free(Set_neg);
			free(Set_pos);
		}
		
		seed = -1;
		for (i=0;i<size;i++) if (!Set[i]){
			len = 0;
			get_all_connected_segments(Segments,size,i,NULL,&len);
			if (len==2){
				seed = i;
				break;
			}
		}
	};		
		
	free(Set);
}

void polygons_from_polyfile(char* name,point2D* Points,int*** Polys,int** Sizes,point2D** Seeds,double** Values,int* number,int* attr_num,int quiet){
	int i,j,ind,ind1,ind2,ind3,ind4,start,next;
	size_t bsize = 0;
	char* Buffer = NULL;
	
	FILE* file = fopen(name,"r");
	if (file==NULL){
		printf("polygons_from_poly_file: could not open file %s -> abort\n",name);
		exit(0);
	}
	
	getline(&Buffer,&bsize,file);
	Buffer[bsize-1] = '\0';
	char* Part = strtok(Buffer," ");
	int nodes = atoi(Part);
	Part = strtok(NULL," ");
	int dim = atoi(Part);
	Part = strtok(NULL," ");
	*attr_num = atoi(Part);
	free(Buffer);
	Buffer = NULL;
	
	// get segments
	getline(&Buffer,&bsize,file);
	Part = strtok(Buffer," ");
	int total_size = atoi(Part);
	int* Segments = (int*)malloc(2*total_size*sizeof(int));
	read_segments_from_polyfile(file,Segments,total_size);
	free(Buffer);
	Buffer = NULL;
	
	// construct polygons
	Polygons_from_segments(Points,Segments,total_size,Polys,Sizes,number,quiet);
	free(Segments);
	
	
	// skip holes
	getline(&Buffer,&bsize,file);
	int h = atoi(strtok(Buffer," \n"));
	free(Buffer);
	Buffer = NULL;
	for (i=0;i<h;i++){
		getline(&Buffer,&bsize,file);
		free(Buffer);
		Buffer = NULL;
	}
	
	// get attributes	
	getline(&Buffer,&bsize,file);
	int a = atoi(strtok(Buffer," \n"));
	free(Buffer);
	Buffer = NULL;
	if (a!=*number){
		printf("inconsistent polyfile: attribute number (%d) does not match polygon number (%d) -> abort\n",a,*number);
		//exit(0);
	}
		
	*Seeds = (point2D*)malloc(a*sizeof(point2D));
	*Values = (double*)malloc(a*(*attr_num)*sizeof(double));
	for (i=0;i<a;i++){
		getline(&Buffer,&bsize,file);
		ind = atoi(strtok(Buffer," \n"));
		(*Seeds)[ind].x = atof(strtok(NULL," \n"));
		(*Seeds)[ind].y = atof(strtok(NULL," \n"));
		for (j=0;j<(*attr_num);j++) (*Values)[ind+a*j] = atof(strtok(NULL," \n"));
		free(Buffer);
		Buffer = NULL;
	}	
}

double* rank4_trace_part_2D(double* C){
	double Tr_left[4];
	double Tr_right[4];
	
	Tr_left[0] = (C[0]+C[12])/2.;
	Tr_left[1] = (C[1]+C[13])/2.;
	Tr_left[2] = (C[2]+C[14])/2.;
	Tr_left[3] = (C[3]+C[15])/2.;
	
	Tr_right[0] = (C[0]+C[3])/2.;
	Tr_right[1] = (C[4]+C[7])/2.;
	Tr_right[2] = (C[8]+C[11])/2.;
	Tr_right[3] = (C[12]+C[15])/2.;
	
	double tr = (C[0]+C[12]+C[3]+C[15])/4.;
	
	double* Res = (double*)malloc(16*sizeof(double));
	
	Res[0] = Tr_left[0]+Tr_right[0]-tr;
	Res[1] = Tr_left[1];
	Res[2] = Tr_left[2];
	Res[3] = Tr_left[3]+Tr_right[0]-tr;
	
	Res[12] = Tr_left[0]+Tr_right[3]-tr;
	Res[13] = Tr_left[1];
	Res[14] = Tr_left[2];
	Res[15] = Tr_left[3]+Tr_right[3]-tr;
	
	Res[4] = Tr_right[1];
	Res[8] = Tr_right[2];
	
	Res[7] = Tr_right[1];
	Res[11] = Tr_right[2];
	
	Res[5] = 0;
	Res[6] = 0;
	Res[9] = 0;
	Res[10] = 0;
	
	return Res;
}

/*double* rank4_projector_2D(double nx,double ny){
	int i,j;
	
	double P2[4];
	
	P2[0] = 1.-ny*ny;
	P2[1] = nx*ny;
	P2[2] = P2[1];
	P2[3] = 1.-nx*nx;
	
	double* P4 = (double*)malloc(16*sizeof(double));
	for (i=0;i<4;i++){
		for (j=0;j<4;j++) P4[4*i+j] = P2[i]*P2[j];
	}
	return P4;
}*/
