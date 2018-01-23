#include "SPH2D.h"

// global variables 

double SPH2D_h;
double SPH2D_R;

int** interpolation_map = NULL;
int** Inv_interpolation_map = NULL;
int T_size;

point2D* Mesh_ref = NULL;

double* Exp_table = NULL;
double table_range;
double arg_fac;
double arg_dx;
double* SPH2D_masses = NULL;
double* SPH2D_weights = NULL;
point2D* SPH2D_points = NULL;
point2D* SPH2D_original = NULL;
double (*Smooth_F)(point2D* X,point2D* Pos,double h);
int** Neighbor_list = NULL;
int* Neighbor_num = NULL;

// Code ///////////////////////////////////////////////

double standard_Gauss2D(point2D* X,point2D* Pos,double h){
	double arg = (X->x-Pos->x)*(X->x-Pos->x)+(X->y-Pos->y)*(X->y-Pos->y);
	arg *= arg_fac;
	if (arg<=table_range) return 0;
	else{
		int i = (int)floor(arg/arg_dx);
		return Exp_table[i];
		/*if (i<T_size) return Exp_table[i];
		else{
			printf("table error: %d\n",i);
			return 0;
		}*/
	}
}

double old_Gauss2D(point2D* X,point2D* Pos,double h){
	double arg = (X->x-Pos->x)*(X->x-Pos->x)+(X->y-Pos->y)*(X->y-Pos->y);
	arg *= -1./(2.*h*h);
	return exp(arg)/(2.*M_PI*h*h);
}

double det_SPH2D(point2D* P1,point2D* P2,point2D* P3){
	return (P1->x)*((P2->y)-(P3->y))
	+(P2->x)*((P3->y)-(P1->y))
	+(P3->x)*((P1->y)-(P2->y));
}

double linear_interpolation_SPH2D(point2D* P1,point2D* P2,point2D* P3
	,double u1,double u2,double u3,point2D* P){
	double detF = det_SPH2D(P1,P2,P3);
	double a = ((u3-u2)*P1->y+(u1-u3)*P2->y+(u2-u1)*P3->y)/detF;
	double b = ((u2-u3)*P1->x+(u3-u1)*P2->x+(u1-u2)*P3->x)/detF;
	double c = ((P2->x*P3->y-P3->x*P2->y)*u1+(P3->x*P1->y-P1->x*P3->y)*u2
		+(P1->x*P2->y-P2->x*P1->y)*u3)/detF;
	return a*P->x+b*P->y+c;
}

void init_SPH2D_from_mesh(point2D* Nodes,double* FEM_weights,int size,double R,double h){
	int i,j;
	point2D* node;
	SPH2D_original = Nodes;
	Neighbor_num = (int*)malloc(size*sizeof(int));
	Neighbor_list = (int**)malloc(size*sizeof(int*));
	SPH2D_h = h;
	SPH2D_R = R;
	for (i=0;i<size;i++){
		Neighbor_list[i] = NULL;
		Neighbor_num[i] = 0;
		node = &(Nodes[i]);
		for (j=0;j<size;j++){
			if (dist(node,&(Nodes[j]))<R){
				Neighbor_list[i]=(int*)realloc(Neighbor_list,(Neighbor_num[i]+1)*sizeof(int));
				Neighbor_list[i][Neighbor_num[i]] = j;
				Neighbor_num[i]++;
			}
		}
	}
	if (SPH2D_points==NULL) SPH2D_points = (point2D*)malloc(size*sizeof(point2D*));
	SPH2D_weights = (double*)malloc(size*sizeof(double));
	for (i=0;i<size;i++){
		SPH2D_points[i] = init_point2D(Nodes[i].x,Nodes[i].y);
		SPH2D_weights[i] = FEM_weights[i];
	}
}

/*double* SPH2D_move(double* Fields,int size,int flow_index,int field_index,double dt){
	int i,j,k;
	point2D* X;
	double sum;
	double total_old = 0;
	for(i=0;i<size;i++){
		SPH2D_points[i].x += dt*Fields[flow_index*size+i];
		SPH2D_points[i].y += dt*Fields[(flow_index+1)*size+i];
		total_old += Fields[field_index*size+i]*SPH2D_weights[i];
	}
	double* Res = zero_vector(size);
	double total_new = 0;
	for(i=0;i<size;i++){
		sum = 0;
		X = &(SPH2D_original[i]);
		for(j=0;j<Neighbor_num[i];j++){
			k = Neighbor_list[i][j];
			sum += Fields[field_index*size+i]*SPH2D_weights[i]*(*Smooth_F)(X,&(SPH2D_points[k]),SPH2D_h);
		}
		Res[i] = sum;
		total_new += Res[i]*SPH2D_weights[i];
	}
	scalar_mult(total_old/total_new,Res,size);
	return Res;
}*/

void set_reference_point(point2D* Ref){
	Mesh_ref = Ref;
}

void set_SPH2D_function(double (*F)(point2D* X,point2D* Pos,double h),double h){
	int i;
	T_size = 100000;
	Smooth_F = F;
	SPH2D_h = h;
	int w = 5;
	arg_fac = -1./(2.*h*h);
	table_range = (double)w*w*h*h*arg_fac;
	//printf("exp table range: %f\n",table_range);
	Exp_table = (double*)malloc(T_size*sizeof(double));
	for (i=0;i<T_size;i++) Exp_table[i] = exp((double)i/T_size*table_range)/(2.*M_PI*h*h);
	arg_dx = (double)table_range/T_size;
}

point2D get_interpolation_v(int ind,point2D* Reg_mesh,point2D* P,double* Vx,double* Vy,int size_x,int size_y,double dx,double dy){
	point2D Res;
	int i3;
	if (Mesh_ref==NULL){
		printf("no reference\n");
		exit(0);
	}
	double ax = (P->x-Mesh_ref->x)/dx;
	double ay = (P->y-Mesh_ref->y)/dy;
	int ix = (int)floor(ax);
	int iy = (int)floor(ay);
	double dax = ax-(double)ix;
	double day = ay-(double)iy;
	if (ix<0 || (ix+1)>=size_x || iy<0 || (iy+1)>=size_y){
		printf("invalid interpolation index: %d:%d\n",ix,iy);
		exit(0);
	}
	int i1 = ix*size_y+iy+1;
	int i2 = (ix+1)*size_y+iy;
	if (day<1.-dax) i3 = ix*size_y+iy; else i3 = (ix+1)*size_y+iy+1;
	point2D* P1 = &Reg_mesh[i1];
	point2D* P2 = &Reg_mesh[i2];
	point2D* P3 = &Reg_mesh[i3];
	double vx1 = Vx[i1];
	double vx2 = Vx[i2];
	double vx3 = Vx[i3];
	double vy1 = Vy[i1];
	double vy2 = Vy[i2];
	double vy3 = Vy[i3];
	int counter = isnan(vx1)+isnan(vx2)+isnan(vx3);
	if (counter!=0){
		switch(counter){
			case 1:
				if (isnan(vx1)){
					vx1 = vx3;
					vy1 = vy3;
				}
				if (isnan(vx2)){
					vx2 = vx1;
					vy2 = vy1;
				}
				if (isnan(vx3)){
					vx3 = vx2;
					vy3 = vy2;
				}
				break;
			case 2:
				if (!isnan(vx1)){
					vx2 = vx1;
					vy2 = vy1;
					vx3 = vx1;
					vy3 = vy1;
				}
				if (!isnan(vx2)){
					vx3 = vx2;
					vy3 = vy2;
					vx1 = vx2;
					vy1 = vy2;
				}
				if (!isnan(vx3)){
					vx1 = vx3;
					vy1 = vy3;
					vx2 = vx3;
					vy2 = vy3;
				}
				break;
			case 3:
				printf("invalid interpolation index: %d:%d\n",ix,iy);
				printf("orig. index: %d\n",ind);
				printf("point: x=%f y=%f\n",P->x,P->y);
				printf("indices: i1=%d i2=%d i3=%d\n",i1,i2,i3);
				printf("x-values: vx1=%f vx2=%f vx3=%f\n",vx1,vx2,vx3);
				print_vector(Vx,size_x*size_y);
				exit(0);
				break;
		}
	}
	Res.x = linear_interpolation_SPH2D(P1,P2,P3,vx1,vx2,vx3,P);
	Res.y = linear_interpolation_SPH2D(P1,P2,P3,vy1,vy2,vy3,P);
	if (isnan(Res.x) || isnan(Res.y)){
		printf("invalid move at index: %d:%d\n",ix,iy);
		exit(0);
	}
	return Res;
}

point2D* regular_move2D(point2D* Reg_mesh,int size_x,int size_y,double dx,double dy,double dt,double* vx,double* vy){
	int i,j,i_x,i_y,mul;					// Achtung x/y-sortierung in Moved_mesh muss stimmen ! 
	double a,dv,tau;
	double eps = 1E-2;
	int n = size_x*size_y;
	point2D New_v;
	point2D Old_v;
	point2D Pred;
	point2D P;
	point2D* New_points = (point2D*)malloc(n*sizeof(point2D));
	//print_point_list2D(Reg_mesh,n,"oldmesh");
	for (i=0;i<n;i++){
		i_x = i / size_y;
		i_y = i % size_y;
		if (!isnan(vx[i])){
			Old_v.x = vx[i];
			Old_v.y = vy[i];
			tau = 2.*dt;
			mul = 1;
			do{
				tau /= 2.;
				mul *= 2;
				Pred.x = Reg_mesh[i].x+Old_v.x*tau;
				Pred.y = Reg_mesh[i].y+Old_v.y*tau;
				New_v = get_interpolation_v(i,Reg_mesh,&Pred,vx,vy,size_x,size_y,dx,dy);
				dv = dist(&New_v,&Old_v);
				if (dv!=0){
					a = dv/vec_abs(&Old_v);
				}
				else a = 0;
			}while(a>eps);
			mul /= 2;
			P.x = Pred.x; // mul-1, da ein Schritt schon gerechnet
			P.y = Pred.y;
			for (j=0;j<mul-1;j++){
				P.x += New_v.x*tau;
				P.y += New_v.y*tau;
				New_v = get_interpolation_v(i,Reg_mesh,&P,vx,vy,size_x,size_y,dx,dy);
			}
			New_points[i].x = P.x;
			New_points[i].y = P.y;
			if (mul>8){
				printf("Warning: very high resolution required at %d: %d\n",i,mul);
			}
		}
		else{
			New_points[i].x = Reg_mesh[i].x;
			New_points[i].y = Reg_mesh[i].y;
		}
	}
	//print_point_list2D(New_points,n,"newmesh");
	return New_points;
}

void apply_regular_move2D(point2D* Reg_mesh,point2D* Moved_mesh,int size_x,int size_y,
 double dx,double dy,double* Field){
	int i,i_x,i_y,j,k,ind;		// Achtung x/y-sortierung in Moved_mesh muss stimmen ! 
	double sum;
	int w = 5;  // nur bei kurzreichweitiger WW !
	int n = size_x*size_y;
	double dA = dx*dy;
	double* Buffer = clone_vector(Field,n);
	//double before = 0;
	//for (i=0;i<n;i++) if (!isnan(Buffer[i])) before += Buffer[i];
	for (i=0;i<n;i++){
		if (!isnan(Buffer[i])){
			i_x = i / size_x;
			i_y = i % size_x;
			sum = 0;
			for (j=i_x-w;j<=i_x+w;j++){
				for (k=i_y-w;k<=i_y+w;k++){
					ind = j*size_x+k;
					if (isnan(Buffer[ind])){
						sum += Buffer[i]*(*Smooth_F)(&Reg_mesh[i],&Reg_mesh[ind],SPH2D_h);
					}
					else{
						sum += Buffer[ind]*(*Smooth_F)(&Reg_mesh[i],&Moved_mesh[ind],SPH2D_h);
					}
				}
			}
			Field[i] = sum*dA;
		}
	}
	//double after = 0;
	//for (i=0;i<n;i++) if (!isnan(Field[i])) after += Field[i];
	//scalar_mult(before/after,Field,n);
	free(Buffer);
}
