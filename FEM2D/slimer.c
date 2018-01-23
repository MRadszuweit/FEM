/* test program */

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <sys/stat.h>
#include "FEM2D.h"
#include "linear_algebra.h"
#include "FEM.h"
#include "File_stuff.h"
#include "itlin.h"
#include "GMRES_Newton.h"

#define free(P) free(P);P=NULL
#define ZERO_MODE -1

extern mesh2D glob_mesh;
extern int FEM_error;

static sparse_matrix* Smith_pattern = NULL;
static double** Fixpoints = NULL;

sparse_matrix3D* System_Nlin = NULL;

sparse_matrix* System_A = NULL;
sparse_matrix* System_B = NULL;
sparse_matrix* System_L = NULL;
sparse_matrix* System_U = NULL;
sparse_matrix* System_BC = NULL;
sparse_matrix* System_E = NULL;
sparse_matrix* RD_L = NULL;
sparse_matrix* RD_U = NULL;

double* Surface_tension = NULL;
double* Random_data = NULL;

static double AREA = 0;
static int Reload = 0;
static int PRINT_INFO = 0;

int** System_cond = NULL;
int* System_pivot_map = NULL;
int* System_LU_map = NULL;
int* System_LU_inv_map = NULL;

/*point2D i_mesh_dh;
int interpolation_size = 0;
int** Interpolation_Map = NULL;
int** Inv_Interpolation_Map = NULL;
point2D* Interpolation_mesh = NULL;*/

char** Var_names_list;

char* Output_name;
char* Mesh_name;
char* Mesh_Dir;
char* Output_Dir;
int step_number;
int data_interval;
double total_time;
double time_step;

int ERROR_MARKER = 0;

double comp_tol;
int GMRES_i_max;
int max_iter;
int GS_max_iter;

double Ka;
double Kb;
double KL;
double KV;
double KE;
double KQ;
double KS;
double KP;
double KD;
double NM;
double Nc;
double Tc;
double Tt;
double KT;
int beta;
double D_b;
double D_c;

//double D_u;
//double D_v;
//double co;
//double eps;
//double a;
//double b;
//double k;

double height = 0.2;
double eta;
double per;
double per0;
double tau_T;

double mu;
double la;
double kappa;
double mu_vis;
double la_vis;
double eta_eff;
double sol_fraction;
double sigma;
double front_width = 0.3;
double front_factor;
double anisotropy = 0.0;

double alpha;

double* table_dc1;
double* table_dc2;
double* table_dc3;
double* table_qa;
double* table_qb;
double* table_Qs;
double* table_Ta;
double* table_Tb;
double dnc;
double inv_Tc;
double inv_Tt;

char* load_matrix;
char* initial;

const double P_physarum = 3.6E4;

const double nc_min = -200;
const double nc_max = 200;
const int table_size = 10000000;

const int n_c = 0; // Achtung: Funktion Dimensionsabhängig
const int n_u = 4;
const int n_v = 6;
const int n_p = 8;
const int n_U = 9;
const int n_s = 11;
/*const int n_ds = 11;
const int n_s = 12;
const int n_r = 13;
const int n_phi = 14;*/


/*!
 * name: main
 * 
 * dsfsdf
 * sdfsdf
 * sdf
 * sdfsdf
 * \todo Dokumentation anfangen
 * \attention Vorsicht beim C schreiben.
 * 
 * @param void : keine Parameter
 * @return Ist eine Zahl 
 */
 
void print_info(){
	PRINT_INFO = 1;
}

void no_info(){
	PRINT_INFO = 0;
}

double my_pow(double x,int n){
	int i;
	double res = 1;
	for (i=0;i<n;i++) res *= x;
	return res;
}

double f_qab(double x,double K){  // Achtung: NM schon als Faktor enthalten !
	double Nk = K*x;
	//double d = 1.+Nk*Nk; old model (wrong)
	//return NM*(Nk+2.*Nk*Nk)/d;
	return 2.*NM*Nk/(1.+Nk);
}

double f_Q(double x){
	//double p = my_pow(KS*x,beta);   old model (wrong)
	//return KQ*p/(1.+p);
	double p = (KS*x)/(1.+KS*x);
	return KQ*my_pow(p,beta);
}

double f_c1(double x,double a){
	//double Na = a*x;
	//double d = 1.+Na*Na;
	//return NM*(-2.*a*Na*Na*(1.+2.*Na)/(d*d)+a*(1.+4.*Na)/d); old model (wrong)
	double d = 1.+a*x;
	return 2.*NM*a/(d*d);
}

double f_c2(double x,double a,double b){
	//return NM*((x*(-a+b-2*(a*a-b*b)*x+a*(a-b)*b*x*x))/((1.+a*a*x*x)*(1.+b*b*x*x)));  old model (wrong)
	return 2.*NM*(1./(1.+a*x)-1./(1.+b*x));
}

double f_Tab(double x,double K){
	double s = K*x/(1.+K*x);    // corected model 
	double q = KP*(1.-s*s);
	return KT*q/(q+KD);
}

double f_b(double* Vars,int k){
	int i = (int)floor((Vars[n_c]-nc_min)*dnc);
	if (i<0 || i>= table_size){
		printf("Bereichsueberschreitung von n_c ! nc=%f\n",Vars[n_c]);
		FEM_error = 1;
		return (1.-Vars[n_c+2])*f_qab(Vars[n_c],Ka)+Vars[n_c+2]*f_qab(Vars[n_c],Kb);
	}
	return (1.-Vars[n_c+2])*table_qa[i]+Vars[n_c+2]*table_qb[i];
}

double slow_fb(double* Vars,int k){
	return (1.-Vars[n_c+2])*f_qab(Vars[n_c],Ka)+Vars[n_c+2]*f_qab(Vars[n_c],Kb);
}

double slow_fp(double* Vars,int k){
	return (1.-Vars[n_c+2])*f_Q(Vars[n_c])-Vars[n_c+2]*KE;
}

double slow_fc(double* Vars,int k){
	return (KL*(Nc-Vars[n_c]-slow_fb(Vars,k))-KV*Vars[n_c]-slow_fp(Vars,k)*f_c2(Vars[n_c],Ka,Kb))/
	 (1.+f_c1(Vars[n_c],Ka)*(1.-Vars[n_c+2])+f_c1(Vars[n_c],Kb)*Vars[n_c+2]);
}

double slow_fTa(double* Vars,int k){
	return (1.-Vars[n_c+2])*f_Tab(Vars[n_c],Ka)+Vars[n_c+2]*f_Tab(Vars[n_c],Kb);
}

double slow_fT(double* Vars,int k){
	return (slow_fTa(Vars,k)-Vars[n_c+3])*inv_Tt;
}

double simple_fT(double* Vars,int k){
	return (-KT*Vars[n_c]-Vars[n_c+3])*inv_Tt;
}

double saturated_fT(double* Vars,int k){
	return (-KT*Vars[n_c]/(1.+Vars[n_c])-Vars[n_c+3])*inv_Tt;
	//double f = (1.+tanh(Ka*(Kb-Vars[n_c])))/2.;
	//return (KT*f-Vars[n_c+3])*inv_Tt;
}

double df_b(double* Vars,int k){
	//return f_c1(Vars[n_c],Ka)*(1.-Vars[n_c+2])+f_c1(Vars[n_c],Kb)*Vars[n_c+2];
	return KL*(Nc-Vars[n_c]-Vars[n_c+1])-KV*Vars[n_c]-slow_fc(Vars,k);
}

double f_p(double* Vars,int k){
	int i = (int)floor((Vars[n_c]-nc_min)*dnc);
	if (i<0 || i>= table_size){
		return (1.-Vars[n_c+2])*f_Q(Vars[n_c])-Vars[n_c+2]*KE;
	}
	return (1.-Vars[n_c+2])*table_Qs[i]-Vars[n_c+2]*KE;
}

double f_c(double* Vars,int k){
	int i = (int)floor((Vars[n_c]-nc_min)*dnc);
	if (i<0 || i>= table_size){
		FEM_error = 1;
		return (KL*(Nc-Vars[n_c]-f_b(Vars,k))-KV*Vars[n_c]-f_p(Vars,k)*f_c2(Vars[n_c],Ka,Kb))/
		 (1.+f_c1(Vars[n_c],Ka)*(1.-Vars[n_c+2])+f_c1(Vars[n_c],Kb)*Vars[n_c+2]);
	}
	return (KL*(Nc-Vars[n_c]-f_b(Vars,k))-KV*Vars[n_c]-f_p(Vars,k)*table_dc3[i])/
	 (1.+table_dc1[i]*(1.-Vars[n_c+2])+table_dc2[i]*Vars[n_c+2]);
}

double f_Ta(double* Vars,int k){
	int i = (int)floor((Vars[n_c]-nc_min)*dnc);
	if (i<0 || i>= table_size){
		FEM_error = 1;
		return (1.-Vars[n_c+2])*f_Tab(Vars[n_c],Ka)+Vars[n_c+2]*f_Tab(Vars[n_c],Kb);
	}
	return (1.-Vars[n_c+2])*table_Ta[i]+Vars[n_c+2]*table_Tb[i];
}

double f_T(double* Vars,int k){
	return (f_Ta(Vars,k)-Vars[n_c+3])*inv_Tt;
}

double brusselator_x(double* Vars,int k){
	return kappa*(Ka+Vars[n_c]*Vars[n_c]*Vars[n_c+2]-(1+Kb)*Vars[n_c]);
}

double brusselator_y(double* Vars,int k){
	return kappa*(-Vars[n_c]*Vars[n_c]*Vars[n_c+2]+Kb*Vars[n_c]);
}

double brusselator_T(double* Vars,int k){
	return kappa*(KT*Vars[n_c+2]-Vars[n_c+3])*inv_Tt;
}

double Const(double* Vars,int k){
	return 1.;
}

double* create_dn_f1_table(int n,double a,double min,double max){
	int i;
	double Na,d,x;
	double* Table = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		x = (double)(min+(max-min)*i/(n-1));
		Na = a*x;
		d = 1.+Na*Na;
		Table[i] = NM*(-2.*a*Na*Na*(1.+2.*Na)/(d*d)+a*(1.+4.*Na)/d);
	}
	return Table;
}

double* create_dn_f2_table(int n,double a,double b,double min,double max){
	int i;
	double x;
	double* Table = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		x = (double)(min+(max-min)*i/(n-1));
		Table[i] = NM*((x*(-a+b-2*(a*a-b*b)*x+a*(a-b)*b*x*x))/((1.+a*a*x*x)*(1.+b*b*x*x)));
	}
	return Table;
}

double* create_q_table(int n,double K,double min,double max){
	int i;
	double Nk,d,x;
	double* Table = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		x = (double)(min+(max-min)*i/(n-1));
		Nk = K*x;
		d = 1.+Nk*Nk;
		Table[i] = NM*(Nk+2.*Nk*Nk)/d;
	}
	return Table;
}

double* create_Q_table(int n,double min,double max){
	int i;
	double p,x;
	double* Table = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		x = (double)(min+(max-min)*i/(n-1));
		p = my_pow(KS*x,beta);
		Table[i] = KQ*p/(1.+p);
	}
	return Table;
}

double* create_T_table(int n,double K,double min,double max){
	int i;
	double s,q,x;
	double* Table = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		x = (double)(min+(max-min)*i/(n-1));
		s = (K*x)*(K*x);
		q = KP*(1.-s/(1.+s));
		Table[i] = KT*q/(q+KD);
	}
	return Table;
}

void create_look_up_tables(){
	/*table_qa = create_q_table(table_size,Ka,nc_min,nc_max);
	table_qb = create_q_table(table_size,Kb,nc_min,nc_max);
	table_Qs = create_Q_table(table_size,nc_min,nc_max);
	table_Ta = create_T_table(table_size,Ka,nc_min,nc_max);
	table_Tb = create_T_table(table_size,Kb,nc_min,nc_max);
	
	table_dc1 = create_dn_f1_table(table_size,Ka,nc_min,nc_max);
	table_dc2 = create_dn_f1_table(table_size,Kb,nc_min,nc_max);
	table_dc3 = create_dn_f2_table(table_size,Ka,Kb,nc_min,nc_max);*/
	
	dnc = (double)(table_size-1)/(nc_max-nc_min);
	inv_Tc = 1./Tc;
	inv_Tt = 1./Tt;
}

void free_look_up_tables(){
	free(table_qa);
	free(table_qb);
	free(table_dc1);
	free(table_dc2);
	free(table_dc3);
	free(table_Qs);
	free(table_Ta);
	free(table_Tb);
}

double init_con(int i_vertex,int var_index){
	double x,y,phi,r;
	int n_mesh = glob_mesh.size;
	x = glob_mesh.Points[i_vertex].x;
	y = glob_mesh.Points[i_vertex].y;
	phi = atan2(y,x);
	r = sqrt(x*x+y*y);
	double h = 5.*mean_edge_len();
	if (var_index==n_s) return 0.;
	if (Fixpoints!=NULL){
		if (var_index==n_c) return (Fixpoints[0][n_c]+Random_data[n_c*n_mesh+i_vertex]);
		/*if (var_index==n_c){
			return Fixpoints[0][n_c]+0.5*exp(-r*r/(2.*h*h));
		}*/
		if (var_index==n_c+1) return Fixpoints[0][n_c+1];
		if (var_index==n_c+2) return Fixpoints[0][n_c+2];
		/*if (var_index==n_c+2){
			if (r<0.5) return 0.65; else 0.28;
		}*/
		if (var_index==n_c+3) return Fixpoints[0][n_c+3];//-KT*0.5*exp(-r*r/(2.*h*h));
	}
	else{
		//double s1 = 160/4;
		//double s2 = 60/4;
		//double d = 0.15*4;
	
		//if (var_index==n_c+2 && Random_data!=NULL) return Random_data[(n_c+2)*n_mesh+i_vertex];
		
		if (var_index==n_c) return (double)1.*exp(-20.*r*r);//+((double)rand()/(RAND_MAX))/100000.;
		//if (var_index==n_c) return (double)(2.+sin(5.*M_PI*x))/3.; //Achtung keine negativen Werte !
	
		//double a1 = x*x+(y-0.5)*(y-0.5);
		//double a2 = x*x+(y+0.5)*(y+0.5);
		//if (var_index==n_c) return 1.*exp(-10.*r*r);//1.*exp(-10.*a1)+1.*exp(-10.*a2);
		//if (var_index==n_c+1) return 8.*exp(-10.*r*r);//8.*exp(-10.*a1)+8.*exp(-10.*a2);
	}
	return 0;
}

double sx_inh(point2D* P){
	double x = P->x;
	double y = P->y;
	//double phi = atan2(y,x);
	double r = sqrt(x*x+y*y);
	return exp(-60*r*r);
}

double FWD_Euler_Smith(double* X,int i,int size){
	switch(i){
		case 0: return X[0]+X[1]-X[3];
		case 1: return (1.-X[2])*f_qab(X[0],Ka)+X[2]*f_qab(X[0],Kb)-X[1];
		case 2: return -time_step*(f_Q(X[0])*(1.-X[2])-KE*X[2])+X[2];
		case 3: return -time_step*(KL*(Nc-X[3])-KV*X[0])+X[3];
		default: return 0;
	}
}

double Smith_F(double* X,int var_index,int size){
	double Y[4] = {0,0,0,0};
	Y[0] = X[0];
	Y[2] = X[1]; 
	switch(var_index){
		case 0: return slow_fc(Y,0);
		case 1: return slow_fp(Y,0);
	}
	return 0;
}

double Smith4_F(double* X,int i,double t,int var_index){
	switch(var_index){
		case 0: return slow_fc(X,i);						// Smith & Saldana
		case 1: return df_b(X,i);
		case 2: return slow_fp(X,i);
		case 3: return slow_fT(X,i);
		
		/*case 0: return 0;									// Bois & Grill
		case 1: return 0;
		case 2: return 0;
		case 3: return 0*saturated_fT(X,i);*/
		
		/*case 0: return brusselator_x(X,i);				// Brusselator
		case 1: return 0;
		case 2: return brusselator_y(X,i);
		case 3: return brusselator_T(X,i);*/
	}
	return 0;
}

double H_0(double* X,int i){
	return X[n_c]*(1.-X[n_c]*X[n_c]-X[n_c+2]*X[n_c+2])-0.5*X[n_c+2];
}

double H_2(double* X,int i){
	return X[n_c+2]*(1.-X[n_c]*X[n_c]-X[n_c+2]*X[n_c+2])+0.5*X[n_c];
}

double H_T(double* X,int i,double t){
	if (t<25.) return (KT*X[n_c+2]-X[n_c+3])*inv_Tt;
	else return -X[n_c+3]*inv_Tt;
}

double Hopf_F(double* X,int i,double t,int var_index){
	double y = 0;
	switch(var_index){
		case 0:y = H_0(X,i);break;
		case 1: y = 0.0;break;
		case 2: y = H_2(X,i);break;
		case 3: y = H_T(X,i,t);break;
	}
	if (isnan(y)){
		printf("Voll extra scheiße !!!!\n");
		int k;
		for (k=0;k<4;k++) printf("%f|",X[k]);
		printf("\n");
		fflush(stdout);
		ERROR_MARKER = 1;
	}
	return y;
}


void Runge_Kutta4_solve(int varnum,int step_num,double* Sol,double t,double dt){
	int i,j;
	int n_mesh = glob_mesh.size;
	double* Y = zero_vector(varnum);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<varnum;j++) Y[j] = Sol[j*n_mesh+i];
		Sol[n_mesh+i] = slow_fb(Y,i); 
	}
	Runge_Kutta4(Sol,varnum*n_mesh,varnum,t,dt,step_num,&Smith4_F);
	free(Y);
}

void Implicit_solve(int varnum,int step_num,double* Sol,double t0,double dt){
	int i,j,k,iter;
	double t,T;
	int n_mesh = glob_mesh.size;
	double old_dt = time_step;
	time_step = dt/step_num;
	double Min[4];
	double Max[4];
	double RMin[4];
	double RMax[4];
	Min[0] = 0.;Max[0] = 10.;
	Min[1] = 0.;Max[1] = 10.;
	Min[2] = 0.;Max[2] = 1.;
	Min[3] = 0.;Max[3] = 20.;
	RMin[0] = -200.;RMax[0] = 200.;
	RMin[1] = -200.;RMax[1] = 200.;
	RMin[2] = -200.;RMax[2] = 200.;
	RMin[3] = -200.;RMax[3] = 200.;
	Newton_first_guess_options(1,varnum,Min,Max);
	Newton_set_range(varnum,RMin,RMax);
	double* X = zero_vector(varnum);
	double* B = zero_vector(varnum);
	
	B[0] = 0;
	B[1] = 0;
	for (i=0;i<n_mesh;i++){
		for (j=0;j<3;j++) X[j] = Sol[j*n_mesh+i];
		X[3] = X[0]+X[1];
		T = Sol[3*n_mesh+i];
		for (k=0;k<step_num;k++){
			B[2] = X[2];
			B[3] = X[3];
			t = t0+k*time_step;
			Newton_set_inhomo(B,varnum);
			iter = Newton_solver(Smith_pattern,X,1000,1E-8,SMALL_SYSTEM,-1);
			T = (T+f_Ta(X,0)*time_step)/(1.+inv_Tt*time_step);  // Achtung: f_Ta falsch ?
			if (iter<0) printf("viele Iterationen bei index %d\n",i);
		}
		for (j=0;j<3;j++) Sol[j*n_mesh+i] = X[j];
		Sol[3*n_mesh+i] = T;
	}
	free(X);
	free(B);
	time_step = old_dt;
}

/*void splitted_GDGL_solve(int varnum,int k,double* Sol,double t,double dt){
	int i;
	double* F_c;
	double* F_b;
	double* F_p;
	double* F_T;
	int n_mesh = glob_mesh.size;
	double* Res = clone_vector(Sol,varnum*n_mesh);
	double ds = (double)dt/k;
	for (i=0;i<k;i++){
		// end global coupling
		F_c = get_function(Res,n_mesh,varnum,n_c,&f_c);   // Achtung: gilt nur wenn n_c = 0 !
		F_b = get_function(Res,n_mesh,varnum,n_c+1,&f_b);
		F_p = get_function(Res,n_mesh,varnum,n_c+2,&f_p);
		F_T = get_function(Res,n_mesh,varnum,n_c+3,&f_T);
		set_zero(Res,(n_c+1)*n_mesh,(n_c+2)*n_mesh);
		vector_add(Res,F_c,ds,varnum*n_mesh);
		vector_add(Res,F_b,1.,varnum*n_mesh);    // set to 1.
		vector_add(Res,F_p,ds,varnum*n_mesh);
		vector_add(Res,F_T,ds,varnum*n_mesh);    
		free(F_c);
		free(F_b);
		free(F_p);
		free(F_T);
	}
	copy_vector_content(Res,Sol,0,0,varnum*n_mesh);
	free(Res);
}*/

double dirichlet(int i,int n,double t,double dt){	
	double x,y,p,r;
	x = glob_mesh.Points[i].x;
	y = glob_mesh.Points[i].y;
	r = sqrt(x*x+y*y);
	p = atan2(y,x);
	
	if (n==n_c)  return (double)0*sin(4*p)/10;  
	if (n==n_c+1)return 0;						
	if (n==n_u)  return 0;						//ux
	if (n==n_u+1)return 0;						//uy
	if (n==n_v)  return 0;						//vx
	if (n==n_v+1)return 0;						//vy
	return 0;
}

double neumann(int i,int n,double t,double dt){
	double x,y,p,r;
	x = glob_mesh.Points[i].x;
	y = glob_mesh.Points[i].y;
	r = sqrt(x*x+y*y);
	p = atan2(y,x);
	
	if (n==n_c)  return 0*D_b*dt;				//u
	if (n==n_c+1)return 0*D_c*dt;				//v
	if (n==n_u)  return 0;						//sx
	if (n==n_u+1)return 0;						//sy
	if (n==n_v)  return 0;						//Vx
	if (n==n_v+1)return 0;						//Vy
	return 0;
}

double robin(int i,int n,double t,double dt){
	double x,y,p,r;
	if (i>=0){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		r = sqrt(x*x+y*y);
		p = atan2(y,x);
		
		if (n==n_c)  return 0*D_b*dt;				//u
		if (n==n_c+1)return 0*D_c*dt;				//v
		if (n==n_u)  return 0;						//sx
		if (n==n_u+1)return 0;						//sy
		if (n==n_v)  return 0;						//Vx
		if (n==n_v+1)return 0;						//Vy
		return 0;
	}else{return alpha;}
}

int cond(int j_vertex,int k_var){
	double x,y,p;
	if (glob_mesh.Is_boundary[j_vertex]>=0){
		x = glob_mesh.Points[j_vertex].x;
		y = glob_mesh.Points[j_vertex].y;
		p = atan2(y,x);
		
		if (k_var==n_p && j_vertex==0) return DIRICHLET;
		if (k_var==n_c)  return NEUMANN;
		if (k_var==n_c+1)return NEUMANN;
		if (k_var==n_c+2)return NEUMANN;
		if (k_var==n_c+3)return NONE;
		if (k_var==n_u)  return DIRICHLET;
		if (k_var==n_u+1)return DIRICHLET;
		if (k_var==n_v)  return DIRICHLET;
		if (k_var==n_v+1)return DIRICHLET;
		if (k_var==n_p)  return NONE;
		if (k_var==n_s) return NONE;
		return NONE;
	}		
	return NONE;
}

double F_mask(double x,double y){
	return y;
}

/*void test_vector(double* Solution,int size,double eps,char* label){
	int i,k;
	for (i=0;i<size;i++) if (Solution[i]<-eps){
		k = i % glob_mesh.size;
		printf("negative concentration: at index %d: c=%f %s\n",k,Solution[i],label);
		exit(0);
	}
}*/

double* global_linear_average_coupling(double* Solution,int var2,int var1,int varnum){
	int i;
	double mean = 0;
	int n_mesh = glob_mesh.size;
	double* P = set_vector_Ui_0_2D(1);
	for (i=0;i<n_mesh;i++) mean += P[i]*Solution[var1*n_mesh+i];
	mean /= get_total_area();
	double* Res = zero_vector(varnum*n_mesh);
	for (i=0;i<n_mesh;i++) Res[var2*n_mesh+i] = mean*P[i];
	free(P);
	return Res;
}

double* global_quadratic_average_coupling(double* Solution,int var_out,int var_in,int var_ave,int varnum){
	int i;
	double mean = 0;
	int n_mesh = glob_mesh.size;
	double* P = set_vector_Ui_0_2D(1);
	for (i=0;i<n_mesh;i++) mean += P[i]*Solution[var_ave*n_mesh+i];
	mean /= get_total_area();
	int old_num = get_var_number();
	set_var_number(varnum);
	sparse_matrix* A = set_matrix_Aij_00_2D(var_out,var_in,&insert_AV);
	double* Res = sparse_mult(A,Solution);
	scalar_mult(mean,Res,n_mesh);
	free(P);
	free_sparse(A);
	set_var_number(old_num);
	return Res;
}

/*void geom_test(){
	int i,j,k;
	double suma,sumb;
	double sum = 0;
	int n_mesh = glob_mesh.size;
	set_var_number(1);
	double* U = zero_vector(n_mesh);
	for (i=0;i<n_mesh;i++) U[i] = 1.;
	sparse_matrix* A = set_matrix_Aij_00_2D(0,0,&insert_AV);
	sparse_matrix* B = set_matrix_Bijk_000(0,0,U,0,&insert_AV);
	for (i=0;i<n_mesh;i++){
		suma = 0;
		sumb = 0;
		for (j=0;j<A->Len[i];j++) suma += A->Values[i][j];
		for (j=0;j<B->Len[i];j++) sumb += B->Values[i][j];
		sum += (suma-sumb)*(suma-sumb)/(suma*suma);
	}
	printf("test matrix: %f\n",sqrt(sum/n_mesh));
	exit(0);
}*/

double test_n(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	return exp(-20.*(x-0.0)*(x-0.0))/10.;
}

double Vx_func(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	double r2 = x*x+y*y;
	double p = atan2(y,x);
	double v = 2.*r2*(1.-r2);
	return 1.;
}

double Vy_func(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	double r2 = x*x+y*y;
	double p = atan2(y,x);
	double v = 2.*r2*(1.-r2);
	return 0*v*cos(p);
}

double Radial_anisotropy(double* X,int k_mesh,int k_index){
	double x = glob_mesh.Points[k_mesh].x;
	double y = glob_mesh.Points[k_mesh].y;
	double r = x*x+y*y;
	double p = atan2(y,x);
	double nx = sin(p);
	double ny = cos(p);
	switch(k_index){
		case 0: return 1.;
		case 1: return 0;
		case 2: return 0;
		case 3: return 1;
		/*case 0: return nx*nx+anisotropy*ny*ny;
		case 1: return (1.-anisotropy)*nx*ny;
		case 2: return (1.-anisotropy)*ny*nx;
		case 3: return ny*ny+anisotropy*nx*nx;*/
	}
	return 0;
}

void RDA_part(double* Solution,double* Old_Solution,double t,double dt){
	static int counter = 0;
	if (dt!=0){
		double r,r0;
		int n_mesh = glob_mesh.size;
		int oldnum = get_var_number2D();
		//double eps_b = 0.01;  // schon hier mit dt multiplizieren !!!
		//int k = floor(dt/(eps_b*Tc));
		
		int N = 4;
		set_var_number2D(N);
		double* Sol = zero_vector(N*n_mesh);
		double* F = zero_vector(N*n_mesh);
		copy_vector_content(Solution,Sol,0,0,N*n_mesh);
		
	//operator splitting
		
		// ODE reaction
		
		//GDGL_solve(N,1,Sol,t,dt);
		Runge_Kutta4_solve(N,5,Sol,t,dt);
		//Euler_step(Sol,N*n_mesh,N,t,dt,&Smith4_F);
		
		/*int i;													// instantanious coupling
		double c;
		for (i=0;i<n_mesh;i++){
			c = Sol[i];
			Sol[(n_c+3)*n_mesh+i]=-KT*c/(1.+c);
		}*/
		
		// advection
		double* C = restrict_vector_by(n_c*n_mesh,(n_c+1)*n_mesh,Sol);
		double* U = restrict_vector_by(n_u*n_mesh,(n_u+2)*n_mesh,Solution);
		double* V = restrict_vector_by(n_v*n_mesh,(n_v+2)*n_mesh,Solution);
		
		sparse_matrix* Adv = FVM_convective_fluxes_2D(C,V,U);
		double* CV = clone_vector(C,n_mesh);
		
		sparse_matrix* E = sparse_identity(n_mesh);
		sparse_add(E,Adv,-dt);
		sparse_matrix* L_Adv = sparse_zero(n_mesh);
		sparse_matrix* U_Adv = sparse_zero(n_mesh);
		incomplete_LU_factorization(E,L_Adv,U_Adv);
		set_solver_mode(ILU_PCG);
		set_system_matrix(E);
		r0 = matrix_residuum(E,CV,C);
		if (r0>0) FEM_solver(C,CV,&matvec,L_Adv,U_Adv);
		free_sparse(E);
		free_sparse(U_Adv);
		free_sparse(L_Adv);
		double diff=(get_total_amount(CV,0)-get_total_amount(C,0))/get_total_amount(C,0);
		if (abs(diff)>1E-5){
			printf("warnig: large total conc. dev. %f\n",diff);
		}
		
		copy_vector_content(CV,Sol,0,0,n_mesh);									//Konvektion an/aus 
		free_sparse(Adv);
		free(CV);
		free(C);
		free(U);
		free(V);
		
		// PDE diffusion
		
		linear_map(F,1.,System_E,Sol);
		set_solver_mode(ILU_PCG);
		set_system_matrix(System_B);
		r0 = matrix_residuum(System_B,Sol,F);
		if (r0>0){
			//solver(F,Sol,&preconlr);
			FEM_solver(F,Sol,&matvec,RD_L,RD_U);
			r = matrix_residuum(System_B,Sol,F);
			if (PRINT_INFO) printf("relative pcg residuum = %f\n",r/r0);
		}
		else if (PRINT_INFO) printf("initial residuum = 0 -> continue...\n");
		copy_vector_content(Sol,Solution,0,0,N*n_mesh);
		
		test_vector(Sol,4*n_mesh,"bad pos 2");
		
		// loeschen bei bestimmten t
		//if (t==4.0) set_solution_to_zero(Solution,0,&F_mask);   // löschen an !
		
		// clean
		free(F);
		free(Sol);
		set_var_number2D(oldnum);
		counter++;
	}
}

double test_ux(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	return sin(2*M_PI*(x+y));
}

double test_uy(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	return -cos(2*M_PI*(x+y));
}
double test_phi(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	return sin(4.*M_PI*(x+y))+cos(3.*M_PI*(x+y));
}

double rectangle(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	if (x<-0.05 || x>0.05) return 0.;
	if (y<-0.05 || y>0.05) return 0.;
	return 1.;
}

double p_surface(double* X,int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	double r = sqrt(x*x+y*y);
	double phi = atan2(y,x);
	return 1.0;
}

double test_F(double* X,int i,int n){
	int j,k,s1,s2;
	double sum = 0;
	if (glob_mesh.Is_boundary[i]<0){			// dirichlet 
		for (j=0;j<System_Nlin->Len[i];j++){
			s1 = System_Nlin->Indices1[i][j];
			s2 = System_Nlin->Indices2[i][j];
			sum -= System_Nlin->Values[i][j]*X[s1]*X[s2];
		}
	}
	for (j=0;j<System_B->Len[i];j++){
			k = System_B->Indices[i][j];
			sum += System_B->Values[i][j]*X[k];
		}
	return sum;
}

double small_F(double* X,int i,int n){
	if (i==0) return (X[0]-0.5)*(X[0]-0.5)+(X[1]-0.25)*(X[1]-0.25)-1.;
	else return (X[0]+0.5)*(X[0]+0.5)+X[1]*X[1]-1.;
}

matrix2D* diffusion_tensor(int i){
	double x,y;
	double D1 = 4.;
	double D2 = 1.;
	x = glob_mesh.Points[i].x;
	y = glob_mesh.Points[i].y;
	double phi = 0*M_PI*sin(100.*x)*sin(100.*y);
	point2D n = init_point2D(cos(phi),sin(phi));
	matrix2D* Dif = unity_matrix2D();
	matrix2D* H = tensor_product(&n,&n);
	mat_mult(Dif,D2);
	mat_add(Dif,H,D1-D2);
	free(H);
	return Dif;
}

/*void AMG_part(double* Solution,double* Old_Solution,double t,double dt){
	int i;
	matrix2D* D;
	int n_mesh = glob_mesh.size;
	double* S = zero_vector(n_mesh);
	double r = AMG_solve(AMG_data,AMG_F,S);
	printf("AMG residuum: %e\n",r);
	for (i=0;i<n_mesh;i++){
		Solution[n_mesh*n_s+i] = S[i];
		D = diffusion_tensor(i);
		Solution[n_mesh*n_p+i] = D->xx;
		free(D);
	}
	free(S);
}*/

double front_factors(int mesh_index,int var_index){
	int o = 4;
	double gamma,new_gamma,frac,new_frac;
	double x = glob_mesh.Points[mesh_index].x;
	double y = glob_mesh.Points[mesh_index].y;
	double r2 = x*x+y*y;
	if (r2>(1.-front_width)*(1.-front_width)){
		double sol = sol_fraction;
		double gel = 1.-sol;
		double new_sol = front_factor*sol;
		double new_gel = 1.-new_sol;
		if (var_index==n_u-o || var_index==n_u+1-o){
			gamma = sol/gel*eta*(1./per0+(1./per-1./per0)*gel);
			new_gamma = new_sol/new_gel*eta*(1./per0+(1./per-1./per0)*new_gel);
			return new_gamma/gamma;
		}
		if (var_index==n_v-o || var_index==n_v+1-o){
			gamma = eta*(1./per0+(1./per-1./per0)*gel);
			new_gamma = eta*(1./per0+(1./per-1./per0)*new_gel);
			return new_gamma/gamma;
		}
		if (var_index==n_p-o){
			frac = sol/gel;
			new_frac = new_sol/new_gel;
			return new_frac/frac;
		}
		return 1.;
	}
	else return 1.;
}

double front_func_elastic(int mesh_index,int var_index){
	int o = 4;
	if (var_index==n_u-o || var_index==n_u+1-o){
		double x = glob_mesh.Points[mesh_index].x;
		double y = glob_mesh.Points[mesh_index].y;
		double r2 = x*x+y*y;
		if (r2>(1.-front_width)*(1.-front_width)){
			return front_factor;
		}
		else return 1.;
	}
	else return 1.;
}

double* surface_tension(int varnum,int equ,int var,int** Cond,double (*F)(double* X,int i)){
	int i;
	int n_mesh = glob_mesh.size;
	int old_num = get_var_number();
	set_var_number(varnum);
	sparse_matrix* A = set_matrix_bound_1_Aij_00_2D(Cond,NEUMANN,equ,var,&insert_A_aV);
	/*//external force
	double x,y,r;
	sparse_matrix* A = sparse_zero(varnum*n_mesh);
	sparse_matrix* Ax= set_matrix_bound_0_Aij_00_2D(Cond,NEUMANN,equ,var,&insert_AV);
	//sparse_add(A,Ax,1.);
	sparse_matrix* Ay= set_matrix_bound_0_Aij_00_2D(Cond,NEUMANN,equ+1,var+1,&insert_AV);
	sparse_add(A,Ay,-1.);
	free_sparse(Ax);
	free_sparse(Ay);*/
	
	double* sigma = zero_vector(n_mesh*varnum);
	for (i=0;i<n_mesh;i++){
		sigma[n_mesh*var+i] = (*F)(NULL,i);
	}
	sparse_multiplication(A,sigma);
	free_sparse(A);
	set_var_number(old_num);
	return sigma;
}

void init_system_matrix(void* Params){
	int i;
	int N = 7;
	int o = 4;
	int M = get_var_number();
	int n_mesh = glob_mesh.size;
	double dt = time_step;
	double sol = sol_fraction;
	double gel = 1.-sol;
	double frac = sol/gel;
	
	double* Factors = (double*)malloc(2*sizeof(double));
	Factors[0] = 1.0;
	Factors[1] = front_factor;
	//Factors[2] = front_factor;
	set_part_attr(Factors);
	
	// per for two different domains
	//double gamma = eta/gel*(1./per0+(1./per-1./per0)*gel); Physarum !
	double gamma = eta/per;
	//double gamma2 = eta/gel*(1./per0+(1./(front_factor*per)-1./per0)*gel);
	
	System_cond = (int**)malloc(n_mesh*sizeof(int*));
	System_A = sparse_zero(N*n_mesh);
	
	set_var_number(N);
	// gel
	set_partition_mode(PARTITION_ON);
	sparse_matrix* E1 = set_matrix_Aij_11_2D(n_u-o,n_U-o,&insert_A_abV_b);
	sparse_matrix* E2 = set_matrix_Aij_11_2D(n_u-o,n_U-o,&insert_A_baV_b);
	sparse_matrix* E3 = set_matrix_Aij_11_2D(n_u-o,n_U-o,&insert_A_bbV_a);
	sparse_matrix* E4 = set_matrix_Aij_00_2D(n_u-o,n_U-o,&insert_AV_a);
	set_partition_mode(PARTITION_OFF);
	sparse_matrix* E5 = set_matrix_Aij_11_2D(n_u-o,n_u-o,&insert_A_bbV_a);
	sparse_matrix* E6 = set_matrix_Aij_11_2D(n_u-o,n_u-o,&insert_A_baV_b);
	sparse_add(System_A,E1,-la);
	sparse_add(System_A,E2,-mu);
	sparse_add(System_A,E3,-mu);
	//sparse_add(System_A,E4,-2.*mu/(height*height));										//z-Scherung an !
	sparse_add(System_A,E5,-mu_vis);												// viskoelastik an !
	//sparse_add(System_A,E6,-mu_vis/3.); 3-dim. effect
	
	//set_partition_mode(PARTITION_ON);
	sparse_matrix* I1 = set_matrix_Aij_00_2D(n_u-o,n_u-o,&insert_AV_a);
	sparse_matrix* I2 = set_matrix_Aij_00_2D(n_u-o,n_v-o,&insert_AV_a);
	//set_partition_mode(PARTITION_OFF);
	sparse_add(System_A,I1,-sol*gamma);
	sparse_add(System_A,I2,sol*gamma);
	sparse_matrix* G1 = set_matrix_Aij_10_2D(n_u-o,n_p-o,&insert_A_aV);
	sparse_add(System_A,G1,1.);
	
	//sol
	sparse_matrix* V1 = set_matrix_Aij_11_2D(n_v-o,n_v-o,&insert_A_bbV_a);
	sparse_add(System_A,V1,-eta_eff);
	sparse_matrix* V2 = set_matrix_Aij_11_2D(n_v-o,n_v-o,&insert_A_baV_b);
	//sparse_add(System_A,V2,-eta_eff/3.); 3-dim. effect
	//set_partition_mode(PARTITION_ON);
	sparse_matrix* J1 = set_matrix_Aij_00_2D(n_v-o,n_u-o,&insert_AV_a);
	sparse_matrix* J2 = set_matrix_Aij_00_2D(n_v-o,n_v-o,&insert_AV_a);
	//set_partition_mode(PARTITION_OFF);
	sparse_add(System_A,J1,gel*gamma);
	sparse_add(System_A,J2,-gel*gamma);
	sparse_matrix* G2 = set_matrix_Aij_10_2D(n_v-o,n_p-o,&insert_A_aV);
	sparse_add(System_A,G2,1.);
	
	//constraint
	sparse_matrix* D1 = set_matrix_Aij_01_2D(n_p-o,n_u-o,&insert_A_aV_a);
	sparse_add(System_A,D1,1.);
	sparse_matrix* D2 = set_matrix_Aij_01_2D(n_p-o,n_v-o,&insert_A_aV_a);
	sparse_add(System_A,D2,frac);
	
	//shift U
	sparse_matrix* C1 = set_matrix_Aij_00_2D(n_U-o,n_U-o,&insert_AV_a);
	sparse_add(System_A,C1,1.+mu);
	sparse_matrix* C2 = set_matrix_Aij_00_2D(n_U-o,n_u-o,&insert_AV_a);
	sparse_add(System_A,C2,-dt*(1+mu));
	
	free_sparse(V1);
	free_sparse(V2);
	free_sparse(E1);
	free_sparse(E2);
	free_sparse(E3);
	free_sparse(E4);
	free_sparse(E5);
	free_sparse(E6);
	free_sparse(I1);
	free_sparse(I2);
	free_sparse(G1);
	free_sparse(G2);
	free_sparse(J1);
	free_sparse(J2);
	free_sparse(D1);
	free_sparse(D2);
	free_sparse(C1);
	free_sparse(C2);
	
	//RD part
	N = 4;
	set_var_number(N);
	System_B = sparse_zero(N*n_mesh);
	
	sparse_matrix* H1 = set_matrix_Aij_00_2D(n_c,n_c,&insert_AV);
	sparse_add(System_B,H1,1.);
	sparse_matrix* H2 = set_matrix_Aij_00_2D(n_c+1,n_c+1,&insert_AV);
	sparse_add(System_B,H2,1.);
	sparse_matrix* H3 = set_matrix_Aij_00_2D(n_c+2,n_c+2,&insert_AV);
	sparse_add(System_B,H3,1.);
	sparse_matrix* H4 = set_matrix_Aij_00_2D(n_c+3,n_c+3,&insert_AV);
	sparse_add(System_B,H4,1.);
	System_E = clone(System_B);
	
	
	//set_partition_mode(PARTITION_ON);
	sparse_matrix* L1 = set_matrix_Aij_11_2D(n_c,n_c,&insert_A_bbV);				// diffusion n_c
	sparse_add(System_B,L1,D_c*dt);
	//set_partition_mode(PARTITION_OFF);
	sparse_matrix* L2 = set_matrix_Aij_11_2D(n_c+2,n_c+2,&insert_A_bbV);			// diffusion phi
	sparse_add(System_B,L2,D_b*dt);
	
	RD_L = sparse_zero(N*n_mesh);
	RD_U = sparse_zero(N*n_mesh);
	incomplete_LU_factorization(System_B,RD_L,RD_U);
	
	free_sparse(H1);
	free_sparse(H2);
	free_sparse(H3);
	free_sparse(H4);
	free_sparse(L1);
	free_sparse(L2);
	
	N = 7;
	set_var_number(N);
	
	//  Boundary conditions
	for (i=0;i<n_mesh;i++){
		System_cond[i] = (int*)malloc(N*sizeof(int));
		System_cond[i][n_p-o] = NONE;
		if (is_boundary(i)>=0){
			/*double y = glob_mesh.Points[i].y;
			if (y>0.0){
				System_cond[i][n_u-o] = NEUMANN;
				System_cond[i][n_u+1-o] = NEUMANN;
			}
			else{
				System_cond[i][n_u-o] = DIRICHLET;
				System_cond[i][n_u+1-o] = DIRICHLET;
			}*/
			
			System_cond[i][n_u-o] = DIRICHLET;
			System_cond[i][n_u+1-o] = DIRICHLET;
			
			//System_cond[i][n_u-o] = ROBIN;					// elastischer Rand
			//System_cond[i][n_u+1-o] = ROBIN;
			
			System_cond[i][n_v-o] = NONE;							// nur wenn du=v*dt
			System_cond[i][n_v+1-o] = NONE;
			System_cond[i][n_U-o] = NONE;
			System_cond[i][n_U+1-o] = NONE;
		}
		else{
			System_cond[i][n_u-o] = NONE;
			System_cond[i][n_u+1-o] = NONE;
			System_cond[i][n_v-o] = NONE;
			System_cond[i][n_v+1-o] = NONE;
			System_cond[i][n_U-o] = NONE;
			System_cond[i][n_U+1-o] = NONE;
		}
	}
	int ind = get_center_index();
	System_cond[ind][n_p-o] = DIRICHLET;									// nur wenn Dirichlet für U
	
	/*sparse_matrix* W = set_matrix_bound_0_Aij_00_2D(System_cond,ROBIN,n_u-o,n_U-o,&insert_AV_a);  // elatischer rand
	sparse_add(System_A,W,sigma);
	free_sparse(W);*/
	
	set_boundary_equation(System_A,NULL,n_v-o,n_u-o,1.); 							// nur wenn du=v*dt am Rand 
	set_boundary_equation(System_A,NULL,n_v+1-o,n_u+1-o,1.); 
	
	/*for (i=0;i<n_mesh;i++){												wenn nur elastische Gleichungen
		reset_row(System_A,(n_v-o)*n_mesh+i);
		insert_sparse(System_A,1.,(n_v-o)*n_mesh+i,(n_v-o)*n_mesh+i);
		
		reset_row(System_A,(n_v+1-o)*n_mesh+i);
		insert_sparse(System_A,1.,(n_v+1-o)*n_mesh+i,(n_v+1-o)*n_mesh+i);
		
		reset_row(System_A,(n_p-o)*n_mesh+i);
		insert_sparse(System_A,1.,(n_p-o)*n_mesh+i,(n_p-o)*n_mesh+i);
	}*/
	
	
	System_BC = FEM_get_dirichlet_correction(System_A,System_cond,1.);
	AREA = get_total_area();
	Smith_pattern = Newton_get_full_pattern(4);
	Newton_set_function(FWD_Euler_Smith);
	//Surface_tension = surface_tension(N,n_u-o,n_u-o,System_cond,&p_surface);				// surface tension
	Surface_tension = zero_vector(N*n_mesh);		// Näherung für den Kreis, da nicht von u abhängig
	
	// Random initial conditions 
	
	Random_data = zero_vector(10*n_mesh);
	FEM_add_noise(Random_data,0.01,10000,n_c);
	
	// suche Fixpunkt

	double min_range[2] = {0,0};
	double max_range[2] = {1,1};
	Newton_first_guess_options(10000,2,min_range,max_range);
	Newton_set_range(2,min_range,max_range);
	sparse_matrix* Pattern = Newton_get_full_pattern(2);
	Newton_set_function(&Smith_F);
	Newton_set_inhomo(NULL,2);
	int rootnum = Newton_solver_mult_Root(Pattern,50,1E-6,SMALL_SYSTEM);
	printf("number of fixpoints: %d\n",rootnum);
	double** Fix = Newton_get_all_roots();
	if (rootnum>0){
		char fix_name[512];
		sprintf(fix_name,"%s/%s",Output_Dir,Output_name);
		FILE* Fix_file = open_file(fix_name,"fixpoints","w");
		Fixpoints = (double**)realloc(Fixpoints,rootnum*sizeof(double*));
		
		//Fix[0][0] =0.78;
		
		for (i=0;i<rootnum;i++){
			//Fixpoints[i] = Fix[i];
			Fixpoints[i] = zero_vector(4);
			Fixpoints[i][0] = Fix[i][0];
			Fixpoints[i][2] = Fix[i][1];
			Fixpoints[i][1] = slow_fb(Fixpoints[i],0);
			Fixpoints[i][3] = slow_fTa(Fixpoints[i],0);				// Smith/Saldana
			
			/*Fixpoints[i][0] = Fix[i][0];
			Fixpoints[i][1] = Fix[i][0];
			Fixpoints[i][2] = Fix[i][0];
			Fixpoints[i][3] = -KT*Fix[i][0]/(1.+Fix[i][0]);*/			// Bois/Grill model
			
			/*Fixpoints[i][0] = Ka;
			Fixpoints[i][1] = 0;
			Fixpoints[i][2] = Kb/Ka;
			Fixpoints[i][3] = KT*Kb/Ka;*/										// Brusselator
			
			printf("fixpoint %d: n_c = %f phi = %f\n",i+1,Fix[i][0],Fix[i][1]);
			fprintf(Fix_file,"%f\t %f\t %f\t %f\n",Fixpoints[i][0],Fixpoints[i][1],
			 Fixpoints[i][2],Fixpoints[i][3]);
			free(Fix[i]);
		}
		fclose(Fix_file);
	}
	else exit(0);
	free(Fix);
	free(Pattern);
	print_sparse(System_A);
	
	char dir[512];
	
	/*double h;
	int i_n = 400;
	double rim = 0.05;
	point2D* Ref = (point2D*)malloc(sizeof(point2D));
	Ref->x = 0;
	Ref->y = 0;
	interpolation_size = i_n*i_n;
	Interpolation_mesh = create_regular_mesh2D(i_n,i_n,rim,Ref);
	i_mesh_dh = get_reg_dist(i_n,i_n,rim);
	if (i_mesh_dh.x<i_mesh_dh.y) h = i_mesh_dh.x; else h = i_mesh_dh.y;
	set_SPH2D_function(&standard_Gauss2D,h);
	set_reference_point(Ref);
	Interpolation_Map = (int**)malloc(interpolation_size*sizeof(int*));
	Inv_Interpolation_Map = (int**)malloc(glob_mesh.size*sizeof(int*));
	for (i=0;i<interpolation_size;i++) Interpolation_Map[i] = NULL;
	for (i=0;i<glob_mesh.size;i++) Inv_Interpolation_Map[i] = NULL;
	create_tri_map(Interpolation_mesh,Interpolation_Map,Inv_Interpolation_Map,interpolation_size,3.);*/
	
	// LU-Zerlegung
	
	//char* path="/home/radszuweit/Daten";
	if (strcmp(load_matrix,"none")!=0){							// Achtung: System_A hängt von dt ab !
		sprintf(dir,"%s/%s/LUinfo",Output_Dir,load_matrix);
		printf("load matrix info from %s ...\n",dir);
		System_L = read_sparse(dir,"L.sparse");
		System_U = read_sparse(dir,"U.sparse");
		int Number[1];
		int Size[1];
		int** Lists = read_lists(dir,"LUmaps.int",Number,Size);
		if (*Number!=3 || *Size!=N*n_mesh){
			printf("Wrong LIinfo-file -> abort\n");
			exit(0);
		}
		System_pivot_map = Lists[0];
		System_LU_map = Lists[1];
		System_LU_inv_map = Lists[2];
		free(Lists);
		sparse_row_permutation(System_A,System_pivot_map);
	}
	else{
		System_pivot_map = get_ID_index_map(System_A->size);
		other_pivot(System_A,NULL,System_pivot_map);
		printf("min diag: %f\n",get_worst_diagonal(System_A));
		
		System_L = sparse_zero(N*n_mesh);
		System_U = sparse_zero(N*n_mesh);
		System_LU_map = zero_int_list(N*n_mesh);
		System_LU_inv_map = zero_int_list(N*n_mesh);
		LU_sparse_block_factorization(System_A,System_L,System_U,System_LU_map,System_LU_inv_map);
		printf("block LU fertig\n");
		sprintf(dir,"%s/%s/LUinfo",Output_Dir,Output_name);
		mkdir(dir,0777);
		write_sparse(System_L,dir,"L.sparse");
		write_sparse(System_U,dir,"U.sparse");
		int* lists[] = {System_pivot_map,System_LU_map,System_LU_inv_map};
		write_lists(dir,"LUmaps.int",lists,3,N*n_mesh);
	}
	sparse_row_permutation(System_BC,System_pivot_map);
	
	//FTI_test(System_L,System_U);
	
	/*sparse_matrix* P = sparse_product(System_L,System_U);
	sparse_matrix* Q = sparse_zero(N*n_mesh);
	map_indices(P,Q,NULL,NULL,System_LU_inv_map);
	sparse_add(Q,System_A,-1.);
	double r = matrix_norm(Q)/matrix_norm(System_A);
	printf("matrix difference %f\n",r);
	free_sparse(P);
	free_sparse(Q);*/
	
	set_var_number(M);
}

void destroy_system_matrix(){
	int i;
	if (System_A!=NULL) free_sparse(System_A);
	if (System_B!=NULL) free_sparse(System_B);
	if (System_L!=NULL) free_sparse(System_L);
	if (System_U!=NULL) free_sparse(System_U);
	//if (System_BC!=NULL) free_sparse(System_BC);
	if (System_E!=NULL) free_sparse(System_E);
	if (RD_L!=NULL) free_sparse(RD_L);
	if (RD_U!=NULL) free_sparse(RD_U);
	if (System_cond!=NULL){
		for (i=0;i<glob_mesh.size;i++) free(System_cond[i]);
		//free(System_cond);
	}
	if (Surface_tension!=NULL) free(Surface_tension);
	if (System_pivot_map!=NULL) free(System_pivot_map);
	if (System_LU_map!=NULL) free(System_LU_map);
	if (System_LU_inv_map!=NULL) free(System_LU_inv_map);
	if (Random_data!=NULL) free(Random_data);
}

double C_func(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	double r2 = x*x+y*y;
	return 1.*exp(-20.*r2);
}

void mechanical_part(double* Solution,double* Old_Solution,double t,double dt){
	int i;
	int n_mesh = glob_mesh.size;  // Achtung dt = 0 für t=0 !
	int N = 7;
	int o = 4;
	int M = get_var_number();
	set_var_number(N);
	double sol = sol_fraction;
	double gel = 1.-sol;
	double* F = zero_vector(N*n_mesh);
	double* T = zero_vector(N*n_mesh);
	double* U = zero_vector(N*n_mesh);
	double* u = zero_vector(N*n_mesh);
	copy_vector_content(Solution,T,(n_c+3)*n_mesh,0,n_mesh);		// wird 4mal kopiert, da Aniso 2x2 matrix
	copy_vector_content(Solution,T,(n_c+3)*n_mesh,n_mesh,n_mesh);
	copy_vector_content(Solution,T,(n_c+3)*n_mesh,2*n_mesh,n_mesh);
	copy_vector_content(Solution,T,(n_c+3)*n_mesh,3*n_mesh,n_mesh);	
	if (Fixpoints!=NULL){
		vector_shift(T,4*n_mesh,-Fixpoints[0][n_c+3]);
		for (i=0;i<n_mesh;i++){
			T[i] *= get_attr(i);
			T[i+n_mesh] *= get_attr(i)*get_attr(i);
			T[i+2*n_mesh] *= get_attr(i)*get_attr(i);
			T[i+3*n_mesh] *= get_attr(i)*get_attr(i);
		}
		vector_shift(T,4*n_mesh,Fixpoints[0][n_c+3]);
	}
	else vector_shift(T,4*n_mesh,-get_mean(T,n_mesh));
	//for (i=0;i<4*n_mesh;i++) if (T[i]<0) T[i] = 0.;						// cutoff für negative T 
	copy_vector_content(Solution,U,n_U*n_mesh,5*n_mesh,2*n_mesh);
	copy_vector_content(Solution,u,n_u*n_mesh,0,2*n_mesh);
	
	double* Aniso = get_matrix_function(Solution,n_mesh,N,0,&Radial_anisotropy);
	vector_pseudo_mult(T,Aniso,N*n_mesh);
	//set_partition_mode(PARTITION_ON);
	sparse_matrix* G1 = set_matrix_Aij_10_2D(0,0,&insert_A_bV_ba);
	//set_partition_mode(PARTITION_OFF);
	
	free(Aniso);
	linear_map(F,1.0,G1,T);
	
	vector_add(F,Surface_tension,sigma,N*n_mesh);
	
	//double a = 0;
	//sparse_matrix* S = stretch_matrix(sqrt(2)*cos(a),sqrt(2)*sin(a),N,0);	// anisotrope Spannung !
	//sparse_multiplication(S,F);
	//free_sparse(S);
	
	sparse_matrix* J1 = set_matrix_Aij_00_2D(5,5,&insert_AV_a);
	linear_map(F,1.+mu,J1,U);
	
	double* Sol = zero_vector(N*n_mesh);
	copy_vector_content(Solution,Sol,n_u*n_mesh,0,N*n_mesh);
	
	delete_dirichlet_components(F,System_cond,N*n_mesh);
	set_boundary_equation(NULL,F,n_v-o,n_u-o,1.);						// nur wenn du=v*dt am Rand
	set_boundary_equation(NULL,F,n_v+1-o,n_u-o,1.);
	vector_permutation(F,System_pivot_map,N*n_mesh);
	linear_map(F,1.,System_BC,Sol);
	double r0 = matrix_residuum(System_A,Sol,F);
	LU_sparse_block_solver(System_L,System_U,System_LU_map,System_LU_inv_map,Sol,F);
	copy_vector_content(Sol,Solution,0,n_u*n_mesh,N*n_mesh);
	  
	double r = matrix_residuum(System_A,Sol,F);
	if (PRINT_INFO) printf("residuum LU: %e\n",r/r0);
	free_sparse(G1);
	free_sparse(J1);
	free(F);
	free(T);
	free(u);
	free(U);
	free(Sol);
	set_var_number(M);
}

void material_part(double* Solution,double* Old_solution,double t,double dt){
	int n_mesh = glob_mesh.size;
	int oldnum = get_var_number2D();
	double* U = restrict_vector_by(n_U*n_mesh,(n_U+2)*n_mesh,Solution);
	double* Sol = restrict_vector_by(n_s*n_mesh,(n_s+1)*n_mesh,Solution);
	
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	set_var_number2D(2);
	sparse_matrix* Div = set_matrix_Aij_01_2D(0,0,&insert_A_aV_a);
	double* divU = sparse_mult(Div,U);
	double* DivU = restrict_vector_by(0,n_mesh,divU);
	free(divU);
	free(U);
	
	set_solver_mode(PCG);
	set_system_matrix(ID);
	double r0 = matrix_residuum(ID,Sol,DivU);
	if (r0>0){
		if (r0>0) FEM_solver(DivU,Sol,&gauss_seidel_precon);
		double r = matrix_residuum(ID,Sol,DivU);
		if (PRINT_INFO) printf("residuum material part: %e\n",r/r0);
		//printf("residuum material part: %e\n",r/r0);
	}
	insert_vector(Solution,Sol,n_s*n_mesh,n_mesh);
	
	free_sparse(ID);
	free_sparse(Div);
	free(Sol);
	free(DivU);
	set_var_number2D(oldnum);
}
	

int** get_var_blocks(int** Len,int blocks){
	int i,j,k;
	Len[0] = (int*)malloc(blocks*sizeof(int*));
	Len[0][0] = 4;
	Len[0][1] = 7;
	Len[0][2] = 1;  
	
	int** Blocks = (int**)malloc(blocks*sizeof(int*));
	k = 0;
	for (i=0;i<blocks;i++){
		Blocks[i] = (int*)malloc(Len[0][i]*sizeof(int));
		for (j=0;j<Len[0][i];j++){
			Blocks[i][j] = k;
			k++;
		}
	}
	void** Solvers = (void**)malloc(blocks*sizeof(void*));
	
	// Löser zuordnen
	Solvers[0] = &RDA_part;
	Solvers[1] = &mechanical_part;
	Solvers[2] = &material_part;
	
	set_partial_solvers(Solvers);
	return Blocks;
}

void change_params(int argc,char* argv[],char** Params,int param_num){
	int i,j;
	int len = 0;
	char** Eq;
	for (i=2;i<argc;i++){
		Eq = split(argv[i],"=",&len);
		if (len!=2) printf("argument %d not valid (%s)-> ignored\n",i,argv[i]);
		else{
			for (j=0;j<param_num;j++){
				if (strcmp(Eq[0],Var_names_list[j])==0){
					if (Params[j]!=NULL) free(Params[j]);
					Params[j] = Eq[1];
					printf("set parameter %s = %s\n",Var_names_list[j],Eq[1]);
					break;
				}
			}
		}
	}
}

void get_params(char* path,int argc,char* argv[],int param_num){
	
	#define SET(A,B) Var_names_list[B] = #A;
	SET(Ka,0)
	SET(Kb,1)
	SET(KL,2)
	SET(KV,3)
	SET(KE,4)
	SET(KQ,5)
	SET(KS,6)
	SET(KP,7)
	SET(KD,8)
	SET(KT,9)
	SET(NM,10)
	SET(Nc,11)
	SET(Tc,12)
	SET(Tt,13)
	SET(D_c,14)
	SET(D_b,15)
	SET(beta,16)
	SET(G,17)
	SET(K,18)
	SET(kappa,19)
	SET(per,20)
	SET(eta,21)
	SET(eta_eff,22)
	SET(alpha,23)
	SET(mu_vis,24)
	SET(la_vis,25)
	SET(sigma,26)
	SET(sol,27)
	SET(front_factor,28)
	SET(output_file_name,29)
	SET(mesh_name,30)
	SET(mesh_direction,31)
	SET(time_step_number,32)
	SET(data_interval,33)
	SET(total_time,34)
	SET(computation_tolerance,35)
	SET(GMRES_base_construction_iterations,36)
	SET(maximum_iteration_number,37)
	SET(Gauss_Seidel_preconditioner_iterations,38)
	SET(Matrix,39);
	SET(Initial,40);
	
	#undef SET
	
	int i;
	double K,G;
	char* name = argv[1];
	if (name!=NULL){
		int size[1] = {0};
		char** Params = read_param_strings(path,name,size);
		if (size[0]!=param_num){
			printf("Falscher Parametersatz mit %d variablen ! -> Abbruch\n",size[0]);
			exit(0);
		}
		change_params(argc,argv,Params,param_num);

		// chemischer Teil
		Ka = atof(Params[0]);
		Kb = atof(Params[1]);
		KL = atof(Params[2]);
		KV = atof(Params[3]);
		KE = atof(Params[4]);
		KQ = atof(Params[5]);
		KS = atof(Params[6]);
		KP = atof(Params[7]);
		KD = atof(Params[8]);
		KT = atof(Params[9]);
		NM = atof(Params[10]);
		Nc = atof(Params[11]);
		Tc = atof(Params[12]);
		Tt = atof(Params[13]);
		D_c = atof(Params[14]);
		D_b = atof(Params[15]);
		beta = atoi(Params[16]);
	
		// mechanischer Teil
		G = atof(Params[17]);
		K = atof(Params[18]);
		kappa = atof(Params[19]);
		per = atof(Params[20]);
		eta = atof(Params[21]);
		eta_eff = atof(Params[22]);
		alpha = atof(Params[23]);
		mu_vis = atof(Params[24]);
		la_vis = atof(Params[25]);
		sigma = atof(Params[26]);
		sol_fraction = atof(Params[27]);
		front_factor = atof(Params[28]);
		
		// geometry
		Output_name = Params[29];
		Mesh_name = Params[30];
		Mesh_Dir = Params[31];
		step_number = atoi(Params[32]);
		data_interval = atoi(Params[33]);
		total_time = atof(Params[34]);
		
		//solver
		comp_tol = atof(Params[35]);
		GMRES_i_max = atoi(Params[36]);
		max_iter = atoi(Params[37]);
		GS_max_iter = atoi(Params[38]);
		load_matrix = Params[39];
		initial = Params[40];
		char p_name[512];
		sprintf(p_name,"%s/%s",Output_Dir,Output_name);
		if (mkdir(p_name,0777)==-1){
			printf("File %s existiert bereits -> abort\n",p_name);
			Reload = 1;
		}
		FILE* Param_file = open_file(p_name,"parameters","w");
		fprintf(Param_file,"chemical parameters:\n");
		for (i=0;i<param_num;i++) fprintf(Param_file,"%s = %s\n",Var_names_list[i],Params[i]);
		fclose(Param_file);
	}
	else{
		// chemischer Teil
		Ka = 1.8;
		Kb = 0.15;
		KL = 0.004*60.;
		KV = 0.08*60.;
		KE = 0.1*60.;
		KQ = 0.2*60.;
		KS = 1.5;
		KP = 0.5*60.;
		KD = 0.2*60.;
		KT = 20000.;
		NM = 10.;
		Nc = 25.;
		Tc = 0.05;
		Tt = 0.1;
		D_c = 0.0001;
		D_b = 0.;
		beta = 4;
	
		// mechanischer Teil
		G = 1.05;
		K = 0.94;   // 0.3
		kappa = 0.0001;
		per = 0.3;
		eta = 0.5;
		eta_eff = 0.5;
		alpha = 1.;
		mu_vis = 1.;
		la_vis = 0.3;
		sigma = 0.;
		sol_fraction = 0.5;
		front_factor = 1.;
	
		// geometry
		Output_name = "debug";
		Mesh_name = "disc_delaunay.1";
		Mesh_Dir = "/home/radszuweit/Daten/meshes2D";
		step_number = 50; // dt ca. 0.1
		data_interval = 5;
		total_time = 1;
		
		//solver
		comp_tol = 1E-8;
		GMRES_i_max = 300;
		max_iter = 2000;
		GS_max_iter = 100;
		load_matrix= "none";
		initial="none";
	}
	mu = G;
	//la=K-2.*G		  //1D
	la = K-G;         //2D
	//la = K-2.*G/3.; //3D
	per0 = height*height/(12.*eta);
	time_step = total_time/(step_number-1);
}

int rb_comp(const void* A,const void* B){
	double* a = (double*)A;
	double* b = (double*)B;
	if ((*a)>(*b)) return 1; else return 0;
}

void rb_free_key(void* A){
}

void rb_free_info(void* I){
}

void rb_print_key(const void* A){
	double* a = (double*)A;
	printf("%f\n",*a);
}

void rb_print_info(void* I){
	int* i = (int*)I;
	printf("%d\n",*i);
}

void test_AMG(){
	int i,j,k;
	int dom_len = 40;
	int N = dom_len*dom_len;
	sparse_matrix* A = Lap_2D_Dirichlet(1.,3.,dom_len);
	printf("Diagonaldominanz A: %f\n",get_diag_dominance(A));
	sparse_matrix* L = sparse_zero(N);
	sparse_matrix* U = sparse_zero(N);
	incomplete_LU_factorization(A,L,U);
	sparse_matrix* Q = get_fast_ILU_preconditioned_matrix(A,L,U,1E-10);
	printf("Diagonaldominanz PrecA: %f\n",get_diag_dominance(Q));
	AMG_set_SOR_relaxation_coeff(1.6);
	double* Sol = zero_vector(N);
	double* F = zero_vector(N);
	for (i=1;i<dom_len-1;i++){
		for (j=1;j<dom_len-1;j++){
			k = i*dom_len+j;
			F[k] = -1./N;
		}
	}
	
	print_sparse(Q);
	double* F1 = clone_vector(F,N);
	double* Sol1 = clone_vector(Sol,N);
	
	L_triang_invert(L,F);
	U_triang_invert(U,F);
	
	AMG_print_info(YES);
	amg_system_info* Data = AMG_setup(Q,1,5,2,4,AGGRESSIVE,STAND_ALONE,STANDARD,NULL,NULL);
	clock_t start = clock();
	double r = AMG_solve(Data,F,Sol);
	clock_t end = clock();
	printf("time: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	printf("residuum: %e\n",r);
	
	
	amg_system_info* Data1 = AMG_setup(A,1,5,2,4,AGGRESSIVE,STAND_ALONE,STANDARD,NULL,NULL);
	r = AMG_solve(Data1,F1,Sol1);
	printf("residuum: %e\n",r);
	double d = vec_dist(Sol,Sol1,N)/euklid_norm(Sol1,N);
	printf("difference: %f\n",d);
	exit(0);
}

int main(int argc, char* argv[]){
	
	//FTI_test(NULL,NULL);
	//test_mem();
	
	//D_u = 0.003;
	//D_u = 0.01; 
	//test_AMG();
	
	/*int i;
	for (i=0;i<argc;i++) printf("%s\n",argv[i]);
	clock_t start = clock();
	while(1){
		if ((double)(clock()-start)/CLOCKS_PER_SEC>5.) break;
	}
	printf("fertig\n");
	exit(0);*/
	
	no_info();
	set_thread_num(1);
	int param_number = 41;
	Var_names_list = (char**)malloc(param_number*sizeof(char*));
	Output_Dir = "/Home/damage/radszuwe/Daten/Physarum";
	get_params(Output_Dir,argc,argv,param_number);
	
	/*int N,M;
	char Path[512];
	sprintf(Path,"%s/%s/data",Mesh_Dir,Output_name);
	printf("%s\n",Path);
	FILE* file = fopen(Path,"r+w");
	get_last_time(file,&M,&N);
	double* Sol = get_last_step_data(file,M,N);
	print_vector(Sol,M*N);
	free(Sol);
	fclose(file);
	exit(0);*/
	
	create_look_up_tables();
	set_init_procedure(&init_system_matrix);
	init_solver(GMRES_i_max,max_iter,GS_max_iter,comp_tol);
	set_var_blocks(&get_var_blocks,3);
	set_initial_equations(&init_con);
	set_boundary_equations(&cond,&dirichlet,&neumann,&robin);
	run_FEM_simulation(Output_name,Output_Dir,Mesh_name,Mesh_Dir,step_number,data_interval,
	 total_time,initial);
	free_look_up_tables();
	destroy_system_matrix();
	
	return 0;
} 

/*

switch between no reaction reactio:
* change:
* 
* - Smith4_F -> functionen ändern
* - mechanical_part -> cutoff T an/aus
* - init_system_matrix -> Fixpoints anpassen
* - init_system_matrix -> Fixpunkt manuell
* - saturatedfT -> c0 einstellen
* - adjust initial noise -> standard: 0.01





*/


/* Brusselator 
 * 
 * - Diffusion von n_c+1 -> n_c+2
 * - Advection aus 
 * - Fixpunkte anpassen
 * - Smith4_F -> Funktionen ändern
 * - Mechanik Funktion -> NULL
 * 
*/


