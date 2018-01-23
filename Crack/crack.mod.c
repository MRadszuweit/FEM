#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "crack.c.opari.inc"
#line 1 "crack.c"
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <sys/stat.h>
#include "linear_algebra.h"
#include "reflective_newton.h"
#include "../FEM2D/FEM2D.h"
#include "../FEM2D/geometry2D.h"
#include "../FEM2D/FEM.h"
#include "umfpack.h"

#ifdef __CUDACC__
#include "cudaLinA.h"
#endif


#define NOCON 0
#define DIRICHLET 1
#define NEUMANN 2
#define ROBIN 3
#define MIXED 4
#define FAILED 0
#define SUCCESS 1

void test_AMG();

// global variables //////////////////////////////////////////////////

extern mesh2D glob_mesh;

static char Output_name[512];
static char Output_dir[512];
static char Mesh_name[512];
static char Mesh_dir[512];

static int dimension = 2;
static int degrees_of_freedom = 3;
static int n_u = 0;
static int n_h = 0;
static double constraint_shift = -0.1;					// prevent initial damage variable to be loacted at the bounds
														// shift down by 10%
static double mu = 1.;
static double lambda = 1.;
static double a = 0.005;
static double kappa = 0.0001;
static double eta_eps = 0.01;
static double energy_tol = 1e-4;

static int time_steps;
static double* Strain_energy = NULL;
static double glob_time = 0;
static double total_time = 1.;
static double dt;
static int sample_interval = 1;
static int global_size = 5;
static int backtracking = YES;

static sparse_matrix3D* Elastic = NULL;
static sparse_matrix3D* Bound_Dirichlet = NULL;
static sparse_matrix3D* Bound_Neumann = NULL;
static sparse_matrix* Regularization = NULL;
static sparse_matrix* Bound_Dirichlet_reg = NULL;
static sparse_matrix* Bound_Neumann_reg = NULL;
static sparse_matrix* Interface = NULL;
static double* Dissipation = NULL;

static double** All_U = NULL;
static double** All_H = NULL;

static double* energy_dissipation = NULL;
static double* energy_constr_lower = NULL;
static double* energy_constr_upper = NULL;
static double gibbs0 = 0;

// Alternate Minimization
static int Alt_info = YES;
static int Alt_max_iter = 400;
static double Alt_eps = 1e-9;

//Reflective Newton
static int* R_Newton_Condition = NULL;
static int R_Newton_max_iter = 200;
static double R_Newton_eps = 1e-8;
static double R_Newton_sigma_l = 0.1;
static double R_Newton_sigma_u = 0.9;
static double R_Newton_rho = 0.1;
static double R_Newton_Delta = 250.;					// sqrt(n)/2 good choice
static double R_Newton_Xi = 1.;
static double R_Newton_trust_lower = 0.6;
static double R_Newton_trust_upper = 1.1;
static double R_Newton_trust_radius_min = 1e-3;
static double R_Newton_trust_radius_max = 1000.;
static double R_Newton_Dir_tol = 1e-6;

// for AMG solver 
static amg_system_info* El_AMG = NULL;
static amg_system_info* Int_AMG = NULL;
static amg_system_info* Diff_AMG = NULL;

static amg_statistics step1_AMG_stat;
static amg_statistics alter_min_stat;

	// BiCGStab
static double BiCGStab_eps = 1e-3;			// residuum reduction in each BiCGStab step
static int BiCGStab_max_iter = 10;			// maximum iteration number

	// AMG

static int AMG_smoothing_iter = 2;

static int step1_AMG_depth = 5;
static int step1_max_iter = 100;			// maximum namber of V cycles in AMG
static double step1_eps = 1e-8;				// residuum reduction for each preconditer step
static double step1_tol = 1e-2;				// elements smaller than this treshold are removed in Prolongation/Restriction

static int step2_AMG_depth = 4;
static int step2_max_iter = 100;			// same for preconditioner in step 2
static double step2_eps = 1e-3;				
static double step2_tol = 1e-1;		

// for direct solver (Multifrontal)

static double* UMFPACK_pattern_options = NULL;
static double* UMFPACK_pattern_info = NULL;
static double* UMFPACK_factor_options = NULL;
static double* UMFPACK_factor_info = NULL;
static double* UMFPACK_solve_options = NULL;
static double* UMFPACK_solve_info = NULL;		

// function declarations /////////////////////////////////////////////////////////

double Bound(int node,int var_index,double t);
double** compute_stress_tensor(double* U,double* H);

// functions /////////////////////////////////////////////////////////////////////



double F_area_disc(point2D* P){
	const double y_min = -1.;
	const double y_max = 1.;
	const double val_min = 0.001;
	const double val_max = .5;
	double alpha = -M_PI/4.5;
	double y = cos(alpha)*P->y-sin(alpha)*P->x;
	return val_min+(val_max-val_min)/(y_max-y_min)*(y-y_min);
}

double F_area_shape1(point2D* P){
	const double s = 0.2;
	const double val_min = 0.05;
	const double val_max = 1.0;
	
	const int n = 7;
	const double l = 1.;
	const double a = 0.2;
	const double h = 0.4;
	
	int i;
	double r,res;
	
	point2D* Points = (point2D*)malloc(n*sizeof(point2D));
	Points[0] = init_point2D(-l,0);
	Points[1] = init_point2D(l,0);		
	Points[2] = init_point2D(l,l);	
	Points[3] = init_point2D(a*l,l);																								
	Points[4] = init_point2D(0,h*l);
	Points[5] = init_point2D(-a*l,l);	
	Points[6] = init_point2D(-l,l);
	res = val_max;
	for (i=0;i<n;i++){		
		if (i==4) r = P->x-Points[4].x; else r = dist(P,&(Points[i]));
		res -= (val_max-val_min)*exp(-r*r/(2.*s*s));
	}
	if (res>=val_min) return res; else return val_min;
}

double F_area_shape2(point2D* P){
	const double s = 0.2;
	const double sx = 0.4;
	const double sy = 0.4;
	const double val_min = 0.05;
	const double val_max = 1.0;
	const double l = 1.;
	int i;
	double r,res;
	point2D p1 = init_point2D(0,0);
	point2D p2 = init_point2D(0,1);
	double r1 = sqrt((P->x-p1.x)*(P->x-p1.x)/(sx*sx)+(P->y-p1.y)*(P->y-p1.y)/(sy*sy));
	double r2 = sqrt((P->x-p2.x)*(P->x-p2.x)/(sx*sx)+(P->y-p2.y)*(P->y-p2.y)/(sy*sy));
	res = val_max;
	
	point2D* Points = (point2D*)malloc(4*sizeof(point2D));
	Points[0] = init_point2D(-l,0);
	Points[1] = init_point2D(l,0);		
	Points[2] = init_point2D(l,l);	
	Points[3] = init_point2D(-l,l);			
	
	
	for (i=0;i<4;i++){		
		r = dist(P,&(Points[i]));
		res -= (val_max-val_min)*exp(-r*r/(2.*s*s));
	}
	res -= (val_max-val_min)*(exp(-r1*r1/2.)+exp(-r2*r2/2.));
	
	if (res>=val_min) return res; else return val_min;
}

double create_2D_mesh(char* name, int bound_nodes){
	point2D* Points = (point2D*)malloc(bound_nodes*sizeof(point2D));
	double area = create_arc(bound_nodes,1.,2.*M_PI,Points);	
	create_poly_file(name,Points,bound_nodes);
	free(Points);
	return area;
}

double create_shape(char* name,int mesh_size){
	const int n = 7;
	const double l = 1.;
	const double a = 0.2;
	const double h = 0.4;
	
	int i;
	point2D* Points = (point2D*)malloc(n*sizeof(point2D));
	
	Points[0] = init_point2D(-l,0);											//	---\  /---
	Points[1] = init_point2D(l,0);											//	|	\/   |
																			//	|		 |
	Points[2] = init_point2D(l,l);											//  |________|
	Points[3] = init_point2D(a*l,l);										
																			
	Points[4] = init_point2D(0,h*l);
	Points[5] = init_point2D(-a*l,l);
	
	Points[6] = init_point2D(-l,l);
	
	create_poly_file(name,Points,n);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape2(char* name,int mesh_size){
	const int n = 4;
	const double l = 1.;
	
	int i;
	point2D* Points = (point2D*)malloc(n*sizeof(point2D));
	
	Points[0] = init_point2D(-l,0);											//	----------
	Points[1] = init_point2D(l,0);											//	|	     |
																			//	|		 |
	Points[2] = init_point2D(l,l);											//  |________|
	
	Points[3] = init_point2D(-l,l);
	
	create_poly_file(name,Points,n);
	return (double)2.*l*l/(1.3*mesh_size);
}

void read_mesh_2D(char* Dir,char* Name){
	int element_number;
	//char Name[256] = "";
	//char** line;
	printf("read mesh from file %s/%s\n\n",Dir,Name);
	element_number = 0;
	read_node_file2D(Dir,Name,&glob_mesh);
	read_element_file2D(Dir,Name,&glob_mesh,NULL,&element_number);
	sort_knots();
	create_look_up_table(&glob_mesh);
	
	int len0,len1;
	int** Borders = (int**)malloc(2*sizeof(int*));
	point2D** Seeds = (point2D**)malloc(sizeof(point2D*));
	int bnum = read_poly_file(Dir,Name,Borders,&len0,&len1,Seeds);
	if (bnum!=0){
		int* Partition = partition2D(Borders[0],*Seeds,len0,bnum);
		if (Partition!=NULL) init_partition(Partition,NULL,bnum);
		enable_partition();
		printf("partitioning enabled\n");
		int i;
		for (i=0;i<bnum;i++) printf("seed %d: %f,%f\n",i,(*Seeds)[i].x,(*Seeds)[i].y);
	}
	else{
		disable_partition();
		printf("partitioning disabled\n");
	}
	
	/*Dual_mesh = (point2D**)malloc(glob_mesh.size*sizeof(point2D*));
	Dual_areas = zero_vector(glob_mesh.size);
	read_voronoi_file2D(path,name,Dual_mesh,Dual_areas,1.);*/
}

FILE* open_global(){
	char Fullname[512];
	char Data[512];
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	sprintf(Data,"%s/%s",Fullname,"global");
	return fopen(Data,"a");
}

FILE* open_local(){
	char Fullname[512];
	char Data[512];
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	sprintf(Data,"%s/%s",Fullname,"local");
	return fopen(Data,"a");
}

void save(double* Sol,double* Globals,int loc_size,int glob_size,double t){
	int i;
	static int counter = 0;
	int n = glob_mesh.size;
	
	// global output
	const int buffsize = 256;
	FILE* Output_global = open_global();
	if (Globals!=NULL && Output_global!=NULL){
		char Buffer[buffsize];
		char* Line = (char*)malloc(buffsize*glob_size*sizeof(char));
		sprintf(Line,"%f\t",t);
		for (i=0;i<glob_size-1;i++){
			sprintf(Buffer,"%e\t",Globals[i]);
			strcat(Line,Buffer);
		}
		sprintf(Buffer,"%e\n",Globals[glob_size-1]);
		strcat(Line,Buffer);
		if (fprintf(Output_global,"%s",Line)<0){
			printf("error writing output file -> abort\n");
			exit(0); 
		}
		free(Line);
	}
	if (Output_global!=NULL){
		fflush(Output_global);
		fclose(Output_global);
	}
	
	//local output
	/*FILE* Output_local = open_local();
	if (Sol!=NULL && Output_local!=NULL){
		fwrite(&t,sizeof(double),1,Output_local);
		for (i=0;i<loc_size;i++){
			fwrite(Sol,sizeof(double),loc_size,Output_local);
		}
	}
	if (Output_local!=NULL){
		fflush(Output_local);
		fclose(Output_local);
	}*/
	counter++;
}

double* compute_divergence(double* X){	
	const int max_iter = 100;
	const double eps = 1e-8;
	
	int m = get_var_number2D();
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	set_var_number2D(2);
	sparse_matrix* Div = set_matrix_Aij_01_2D(0,0,&insert_A_aV_a);
	double* Div2 = sparse_mult(Div,X);
	double* Div1 = restrict_vector_by(0,n,Div2);
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(ID,L,U);
	PCG_default(ID,L,U);
	double* Res = zero_vector(n);
	
	double r0 = matrix_residuum(ID,Res,Div1);
	if (r0>0){
		if (r0>0) pcg(Res,Div1,n,eps,max_iter);
		double r = matrix_residuum(ID,Res,Div1);		
		printf("residuum divergence: %e\n",r/r0);
	}
	
	set_var_number2D(m);
	free_sparse(ID);
	free_sparse(Div);
	free_sparse(L);
	free_sparse(U);
	free(Div1);
	free(Div2);
	return Res;
}

int multifrontal_solver(sparse_matrix* A,double* X,double* b){
	const int system_type = UMFPACK_A; 														 // default: solves Ax=b
	int status;
	int n = A->size;
	void* Pattern_info = NULL;
	void* Factor_info = NULL;
	int* A_col = NULL;
	int* A_ind = NULL;
	double* A_val = NULL;	
	int el_number = 0;
	
	convert_sparse_to_UMFPACK(A,n,&A_col,&A_ind,&A_val,&el_number);
	status = umfpack_di_symbolic(n,n,A_col,A_ind,A_val,&Pattern_info,UMFPACK_pattern_options,UMFPACK_pattern_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_ERROR_invalid_matrix: printf("UMFPACK symbolic: invalid matrix\n");break;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK symbolic: memory insufficient\n");break;
			case UMFPACK_ERROR_internal_error: printf("UMFPACK symbolic: internal error\n");break;
		}
		exit(0);
	}
	
	status = umfpack_di_numeric(A_col,A_ind,A_val,Pattern_info,&Factor_info,UMFPACK_factor_options,UMFPACK_factor_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK numeric: singular matrix\n");
				umfpack_di_free_symbolic(&Pattern_info);	
				free(A_col);
				free(A_ind);
				free(A_val);
				return 0;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK numeric: memory insufficient\n");break;
			case UMFPACK_ERROR_invalid_Symbolic_object: printf("UMFPACK numeric: pattern info invalid\n");break;	
			case UMFPACK_ERROR_different_pattern: printf("UMFPACK numeric: pattern info has changed\n");break;			
		}
		exit(0);
	}
	
	status = umfpack_di_solve(system_type,A_col,A_ind,A_val,X,b,Factor_info,UMFPACK_solve_options,UMFPACK_solve_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK solve: singular matrix\n");
				umfpack_di_free_symbolic(&Pattern_info);
				umfpack_di_free_numeric(&Factor_info);
				free(A_col);
				free(A_ind);
				free(A_val);
				return 0;				
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK solve: memory insufficient\n");break;
			case UMFPACK_ERROR_invalid_Numeric_object: printf("UMFPACK solve: factor info invalid\n");break;	
			case UMFPACK_ERROR_invalid_system: printf("UMFPACK solve: invalid system\n");break;			
		}
		exit(0);
	}
	
	umfpack_di_free_symbolic(&Pattern_info);
	umfpack_di_free_numeric(&Factor_info);
	free(A_col);
	free(A_ind);
	free(A_val);
	return 1;
}


double* compute_scalar_function(double* X,int print_info,double eps){	
	const int max_iter = 100;
	int m = get_var_number2D();
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(ID,L,U);
	PCG_default(ID,L,U);
	double* Res = zero_vector(n);
	// solve 
	double r0 = matrix_residuum(ID,Res,X);
	if (r0>0){
		if (r0>0) pcg(Res,X,n,eps,max_iter);
		double r = matrix_residuum(ID,Res,X);		
		if (print_info) printf("residuum function: %e\n",r/r0);
	}
	// clean
	free_sparse(ID);
	free_sparse(L);
	free_sparse(U);
	set_var_number2D(m);
	return Res;
}

double full_functional(double* X,int n){
	int i;
	double res;
	int N = Interface->size;												// only if damage is onedimensional
	double* U = &(X[0]);
	double* H = &(X[dimension*N]);
	
	// employ Dirichlet conditions
	for (i=0;i<N;i++){
		if (R_Newton_Condition[i]==DIRICHLET) U[i] = Bound(i,0,glob_time);
		if (R_Newton_Condition[i+N]==DIRICHLET) U[i+N] = Bound(i,1,glob_time);
		if (R_Newton_Condition[i+dimension*N]==DIRICHLET) H[i] = Bound(i,dimension,glob_time);
	}
	
	double* S = sparse3D_bilinear(Elastic,U,U);
	linear_map(S,1.,Interface,H);
	vector_add(S,Dissipation,-2.,N);
	res = (scalar(H,S,N)+sparse_bilinear(U,Regularization,U))/2.;
	
	free(S);
	return res;
}

double* full_gradient(double* X){
	int i;
	int n = Interface->size;
	double* G = zero_vector((dimension+1)*n);								// only if damage is onedimensional
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	
	double* GH = sparse3D_bilinear(Elastic,U,U);
	scalar_mult((double)1/2,GH,n);
	linear_map(GH,1.,Interface,H);
	
	vector_add(GH,Dissipation,-1.,n);
	sparse_matrix* A2 = sparse3D_vB(Elastic,H,n*dimension);
	sparse_add(A2,Regularization,1.);
	double* GU = sparse_mult(A2,U);
	copy_vector_content(GU,G,0,0,dimension*n);
	copy_vector_content(GH,G,0,n*dimension,n);
		
	// employ Dirichlet conditions
	for (i=0;i<n;i++){
		if (R_Newton_Condition[i]==DIRICHLET) G[i] = X[i]-Bound(i,0,glob_time);
		if (R_Newton_Condition[i+n]==DIRICHLET) G[i+n] = X[i+n]-Bound(i,1,glob_time);
		if (R_Newton_Condition[i+dimension*n]==DIRICHLET) G[i+dimension*n] = X[i+dimension*n]-Bound(i,dimension,glob_time);
	}

	free_sparse(A2);
	free(GH);
	free(GU);
	return G;
}

sparse_matrix* full_Hesse(double* X){
	int i;
	int n = Interface->size;
	int N = (dimension+1)*n;
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	
	sparse_matrix* Huu = sparse3D_vB(Elastic,H,dimension*n);
	sparse_add(Huu,Regularization,1.);
	enlarge_matrix(Huu,n,N,0,0);
	
	sparse_matrix* Hhu = sparse3D_Bv_symmetric(Elastic,U);
	sparse_matrix* Huh = get_transpose(Hhu,dimension*n);
	enlarge_matrix(Hhu,n,N,dimension,0);
	enlarge_matrix(Huh,n,N,0,dimension);
	sparse_add(Huu,Hhu,1.);
	sparse_add(Huu,Huh,1.);
	
	sparse_matrix* Hhh = clone(Interface);
	enlarge_matrix(Hhh,n,N,dimension,dimension);
	sparse_add(Huu,Hhh,1.);
	
	for (i=0;i<N;i++) if (R_Newton_Condition[i]==DIRICHLET){
		reset_row(Huu,i);
		insert_sparse(Huu,1.,i,i);
	}
	
	free_sparse(Hhu);
	free_sparse(Huh);
	free_sparse(Hhh);	
	
	return Huu;
}

double damage_functional(double* X,int n){
	if (Strain_energy!=NULL){
		int i;
		double res;
		
		// employ zero Dirichlet conditions
		double* H = clone_vector(X,n);
		for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) H[i] = 0;
		
		double* S = clone_vector(Strain_energy,n);
		linear_map(S,1.,Interface,H);
		vector_add(S,Dissipation,-2.,n);
		res = scalar(H,S,n)/2.;
		free(H);
		free(S);
		return res;
	}
	else{
		printf("error: no strain energy set -> abort\n");
		exit(0);
		return 0;
	}
}

double* damage_gradient(double* X){
	if (Strain_energy!=NULL){
		int i;
		int n = Interface->size;
		double* G = sparse_mult(Interface,X);
		vector_add(G,Strain_energy,(double)1/2,n);
		vector_add(G,Dissipation,-1.,n);
		
		// employ zero Dirichlet conditions
		for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) G[i] = X[i];
		
		return G;
	}
	else{
		printf("error: no strain energy set -> abort\n");
		exit(0);
		return NULL;
	}
}



double U_bound(int node,int var_index,double t){							// set values for dirichlet conditions
	double res,x,y,b;
	double b_min = 0.0;
	double b_max = 0.08;													// deformation max 10%
	switch(var_index){
		case 0: 
			x = glob_mesh.Points[node].x;
			y = glob_mesh.Points[node].y;
			b = b_min+(b_max-b_min)*(t/total_time);							// linear increase in time
			//res = (x>=0 ? b : -b);										// two sided x stretch
			res = (x>0 ? 2.*b*(y-0.5) : -2.*b*(y-0.5));						// bending 
			//res = (x>0 ? b : 0);											// right sided x stretch	
			//res = (x+1.)*b/2.;											// scaling x
			break;
		case 1:
			res = 0;
			break;
		default: res = 0;												
	}
	return res;
}

double U_bound_force(int node,int var_index,double t){
	double res,x,y;
	x = glob_mesh.Points[node].x;
	y = glob_mesh.Points[node].y;
	switch(var_index){
		case 0:
			res = 0;
			break;
		case 1:
			res = 0;
			break;			
	}
	return res;
}

double Bound(int node,int var_index,double t){		
	if (var_index==dimension) return 0;   									// Attention: Only when dirichlet=0 for damage variable !
	else return U_bound(node,var_index,t);
}

int* get_U_condition_list(){
	const double tol = 1e-10;
	int i;
	double x,y;
	int d = glob_mesh.size;
	int* Res = zero_int_list(d);
	for (i=0;i<d;i++){
		if (is_boundary(i)>=0){
			x = glob_mesh.Points[i].x;
			y = glob_mesh.Points[i].y;
			//if (fabs(x)>0.8) Res[i] = DIRICHLET; else Res[i] = NEUMANN;				// for disc
			if (fabs(fabs(x)-1.)<tol) Res[i] = DIRICHLET; else Res[i] = NEUMANN;		// shape 1
			//Res[i] = DIRICHLET;
		}
		else Res[i] = NOCON;
	}
	return Res;
}

void set_dirichlet_BC(sparse_matrix* A,double* F,int* List,double (*Bound_func)(int node,int var_index, double t)){
	int i,j;
	int d = glob_mesh.size;
	int deg = A->size / d;
	for (i=0;i<d;i++) if (List[i]==DIRICHLET){
		for (j=0;j<deg;j++){
			if (F!=NULL) F[i+j*d] = (*Bound_func)(i,j,glob_time);
			reset_row(A,i+j*d);
			insert_sparse(A,1.,i+j*d,i+j*d);
		}
	}
}

sparse_matrix* merge_boundaries_matrix(sparse_matrix* Dirichlet,sparse_matrix* Neumann,int* Conditions){
	int i,j,k,l;
	sparse_matrix* A;
	int n = Dirichlet->size;
	sparse_matrix* Res = clone(Dirichlet);
	
	/*sparse_matrix* Res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	Res->Indices = (int**)malloc(n*sizeof(int*));
	Res->Values = (double**)malloc(n*sizeof(double*));
	Res->size = n;
	Res->Len = zero_int_list(n);*/
	
	for (i=0;i<n;i++){
		j = i % glob_mesh.size;
		switch(Conditions[j]){
			case NEUMANN: 
				reset_row(Res,i);
				for (k=0;k<Neumann->Len[i];k++) insert_sparse(Res,Neumann->Values[i][k],i,Neumann->Indices[i][k]);
				break;
			case NOCON: 
				if (near_boundary(j)){
					for (k=0;k<glob_mesh.Sizes[j];k++){
						l = glob_mesh.Connections[j][k];
						if (Conditions[l]==NEUMANN){
							reset_row(Res,i);
							break;
						}
						
					}					
				}
				break;
			
		}
		/*Res->Len[i] = A->Len[i];
		Res->Indices[i] = clone_list(A->Indices[i],A->Len[i]);
		Res->Values[i] = clone_vector(A->Values[i],A->Len[i]);*/
	}
	return Res;
}

double* merge_boundaries_vector(double* U,int* Conditions,int size,double t,double (*Bound_force)(int node,int var_index, double t)){
	int i,j,k;
	double* Res = clone_vector(U,size);
	for (i=0;i<size;i++){
		j = i % glob_mesh.size;
		k = i / glob_mesh.size;
		if (Conditions[j]==NEUMANN) Res[i] = (*Bound_force)(j,k,t);
	}
	return Res;
}

double** compute_stress_tensor(double* U,double* H){
	const double eps = 1e-11;
	double* X;
	double* Y;
	sparse_matrix* A2_rowx;
	sparse_matrix* A2_rowy;
	sparse_matrix3D* B3_rowx;
	sparse_matrix3D* B3_rowy;
	int n = glob_mesh.size;
	double* T_rowx = zero_vector(2*n);
	double* T_rowy = zero_vector(2*n);
	
	set_var_number2D(dimension);
	if (H!=NULL){
		B3_rowx = set_matrix_Bijk_001(0,0,0,&insert_B_cWV_cEab_rowx);
		B3_rowy = set_matrix_Bijk_001(0,0,0,&insert_B_cWV_cEab_rowy);
		X = sparse3D_Bvv(B3_rowx,H,U);
		Y = sparse3D_Bvv(B3_rowy,H,U);
		vector_add(T_rowx,X,lambda,2*n);
		vector_add(T_rowy,Y,lambda,2*n);
		free_sparse3D(B3_rowx);
		free_sparse3D(B3_rowy);
		free(X);
		free(Y);
	
		B3_rowx = set_matrix_Bijk_001(0,0,0,&insert_B_aWV_b_rowx);
		B3_rowy = set_matrix_Bijk_001(0,0,0,&insert_B_aWV_b_rowy);
		X = sparse3D_Bvv(B3_rowx,H,U);
		Y = sparse3D_Bvv(B3_rowy,H,U);
		vector_add(T_rowx,X,mu,2*n);
		vector_add(T_rowy,Y,mu,2*n);
		free_sparse3D(B3_rowx);
		free_sparse3D(B3_rowy);
		free(X);
		free(Y);
	
		B3_rowx = set_matrix_Bijk_001(0,0,0,&insert_B_bWV_a_rowx);
		B3_rowy = set_matrix_Bijk_001(0,0,0,&insert_B_bWV_a_rowy);
		X = sparse3D_Bvv(B3_rowx,H,U);
		Y = sparse3D_Bvv(B3_rowy,H,U);
		vector_add(T_rowx,X,mu,2*n);
		vector_add(T_rowy,Y,mu,2*n);
		free_sparse3D(B3_rowx);
		free_sparse3D(B3_rowy);
		free(X);
		free(Y);
	}
	
	A2_rowx = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_x);
	A2_rowy = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_y);
	linear_map(T_rowx,eta_eps*lambda,A2_rowx,U);
	linear_map(T_rowy,eta_eps*lambda,A2_rowy,U);
	free_sparse(A2_rowx);
	free_sparse(A2_rowy);
	
	A2_rowx = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_x);
	A2_rowy = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_y);
	linear_map(T_rowx,eta_eps*mu,A2_rowx,U);
	linear_map(T_rowy,eta_eps*mu,A2_rowy,U);
	free_sparse(A2_rowx);
	free_sparse(A2_rowy);
	
	A2_rowx = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_x);
	A2_rowy = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_y);
	linear_map(T_rowx,eta_eps*mu,A2_rowx,U);
	linear_map(T_rowy,eta_eps*mu,A2_rowy,U);	
	free_sparse(A2_rowx);
	free_sparse(A2_rowy);
	
	double** Res = (double**)malloc(4*sizeof(double*));
	Res[0] = compute_scalar_function(&(T_rowx[0]),NO,eps);
	Res[1] = compute_scalar_function(&(T_rowx[n]),NO,eps);
	Res[2] = compute_scalar_function(&(T_rowy[0]),NO,eps);
	Res[3] = compute_scalar_function(&(T_rowy[n]),NO,eps);
	
	free(T_rowx);
	free(T_rowy);
	
	return Res;
}


double Gibbs_energy(double* U,double* H){
	int n = glob_mesh.size;
	double* Nonregularized = sparse3D_bilinear(Elastic,U,U);
	double g = scalar(H,Nonregularized,n)/2.;
	g += sparse_bilinear(U,Regularization,U)/2.;
	g += sparse_bilinear(H,Interface,H)/2.;
	free(Nonregularized);
	return g;
}

double Interface_energy(double* H){
	return sparse_bilinear(H,Interface,H)/2.;
}

double Damage_dissipation(double* H_prev,double* H){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double d = -a*scalar(V,H,n);
	if (H_prev!=NULL) d += a*scalar(V,H_prev,n);
	free(V);
	return d;
}

double External_work(double* U_prev,double* U,double* H){
	double de;
	int n = glob_mesh.size;
	double* dU = clone_vector(U,dimension*n);
	vector_add(dU,U_prev,-1.,dimension*n);
	
	double* E1 = sparse3D_bilinear(Elastic,U,dU);
	double* E2 = sparse3D_bilinear(Elastic,U_prev,dU);
	de = (scalar(H,E1,n)+scalar(H,E2,n))/2.;
	de += (sparse_bilinear(U,Regularization,dU)+sparse_bilinear(U_prev,Regularization,dU))/2.;
	
	free(E1);
	free(E2);
	free(dU);
	return de;
}

double* vector_along_path(int* List,int size,double* X){
	int i,j,k,l,m;
	int n = glob_mesh.size;
	double* Res = zero_vector(dimension*n);
	for (i=0;i<size;i++){
		k = List[i];
		for (j=0;j<glob_mesh.Sizes[k];j++){
			l = glob_mesh.Connections[k][j];
			if (near_boundary(l)){
				for (m=0;m<dimension;m++) Res[l+m*n] = X[l+m*n];
			}			
		}
		for (m=0;m<dimension;m++) Res[k+m*n] = X[k+m*n];
	}
	return Res;
}

double Boundary_work(double* U_prev,double* U,double* H){		//         Attention: add work from inner forces
	int i,j;
	double res;
	int n = glob_mesh.size;
	set_var_number2D(dimension);
	double* dU = clone_vector(U,dimension*n);
	vector_add(dU,U_prev,-1.,dimension*n);
	sparse_matrix* Dirichlet = sparse3D_Bv(Bound_Dirichlet,H);
	sparse_add(Dirichlet,Bound_Dirichlet_reg,1.);
	sparse_matrix* DT = get_transpose(Dirichlet,dimension*n);
	int* Cond = get_U_condition_list();
	
	double* Strain = zero_vector(dimension*n);
	linear_map(Strain,(double)1/2,DT,U);
	linear_map(Strain,(double)1/2,DT,U_prev);
	
	// replace Neumann Boundaries with force function
	for(i=0;i<n;i++){
		if (Cond[i]==NEUMANN){
			for (j=0;j<dimension;j++) Strain[i+j*n] = (U_bound_force(i,j,glob_time)+U_bound_force(i,j,glob_time-dt))/2.;	
		}
	}
	res = scalar(Strain,dU,dimension*n);
	
	free_sparse(Dirichlet);
	free_sparse(DT);
	free(dU);
	free(Strain);
	free(Cond);
	return res;
}

void UMFPACK_settings(){
	// pattern
	
	// factorization
	
	// solve

}

void setup(){
	FILE* Output_global = NULL;
	FILE* Output_local = NULL;

	// general settings
	set_thread_num(1);
	UMFPACK_settings();
	
	// init output
	char Fullname[512];
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	if (mkdir(Fullname,0777)==-1){
		printf("File %s existiert bereits -> abort\n",Fullname);
		exit(0);
	}
	
	Output_global = open_global();
	Output_local = open_local();
	if (Output_local==NULL || Output_global==NULL){
		printf("could not create output files -> abort\n");
		exit(0);
	}
	
	// load mesh
	read_mesh_2D(Mesh_dir,Mesh_name);	
			
	// set basic matrices
	if (dimension==2){
		
		// elastic energy
		set_var_number2D(1);
		Elastic = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_V_aB_abV_b);					// Achtung: funktioniert nur wenn n_h=0
		sparse3D_scalar_mult(Elastic,lambda);
		sparse_matrix3D* B2 = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_V_aB_baV_b);		
		sparse3D_scalar_mult(B2,mu);
		sparse_matrix3D* B3 = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_B_aaV_bV_b);
		sparse3D_scalar_mult(B3,mu);
		sparse3D_add(Elastic,B2);
		sparse3D_add(Elastic,B3);
		free_sparse3D(B2);
		free_sparse3D(B3);
		
		set_var_number2D(dimension);
		Regularization = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_abV_b);
		scalar_sparse_mult(lambda*eta_eps,Regularization);
		sparse_matrix* A2 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_baV_b);
		sparse_matrix* A3 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_bbV_a);
		sparse_add(Regularization,A2,mu*eta_eps);
		sparse_add(Regularization,A3,mu*eta_eps);
		free_sparse(A2);
		free_sparse(A3);		
		
		Bound_Dirichlet_reg = set_matrix_bound_2_Aij_10_2D(n_u,n_u,&insert_A_abV_b);
		scalar_sparse_mult(lambda*eta_eps,Bound_Dirichlet_reg);
		sparse_matrix* Bound_A2 = set_matrix_bound_2_Aij_10_2D(n_u,n_u,&insert_A_baV_b);
		sparse_matrix* Bound_A3 = set_matrix_bound_2_Aij_10_2D(n_u,n_u,&insert_A_bbV_a);
		sparse_add(Bound_Dirichlet_reg,Bound_A2,mu*eta_eps);
		sparse_add(Bound_Dirichlet_reg,Bound_A3,mu*eta_eps);
		free_sparse(Bound_A2);
		free_sparse(Bound_A3);
		
		Bound_Dirichlet = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_abV_bV_aW);				
		sparse3D_scalar_mult(Bound_Dirichlet,lambda);
		sparse_matrix3D* Bound_B2 = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_baV_bV_aW);		
		sparse3D_scalar_mult(Bound_B2,mu);
		sparse_matrix3D* Bound_B3 = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_aaV_bV_bW);
		sparse3D_scalar_mult(Bound_B3,mu);
		sparse3D_add(Bound_Dirichlet,Bound_B2);
		sparse3D_add(Bound_Dirichlet,Bound_B3);
		free_sparse3D(Bound_B2);
		free_sparse3D(Bound_B3);
		
		Bound_Neumann_reg = set_matrix_bound_0_Aij_00_2D(n_u,n_u,&insert_AV_a);
		Bound_Neumann = set_matrix_bound_0_Bijk_000_2D(n_h,n_u,n_u,&insert_BWV_aV_a);
		
		// interface energy
		set_var_number2D(1);
		Interface = set_matrix_Aij_11_2D(n_h,n_h,&insert_A_bbV);		
		scalar_sparse_mult(kappa,Interface);
		Dissipation = set_vector_Ui_0_2D(1);
		scalar_mult(a,Dissipation,Interface->size);		
	}
	if (dimension==3){
		printf("dimension 3 not implemented yet ...");
		exit(0);
	}
	
	// init statistics
	Alt_info = NO;
	RN_print_info(NO);
	BI_print_info(YES);
	step1_AMG_stat = AMG_init_stat("/Home/damage/radszuwe/Daten/AMG_step1");
	BI_set_stat("/Home/damage/radszuwe/Daten/AMG_step2","/Home/damage/radszuwe/Daten/BiCGStab_step2");
	RN_init_statistics();
	alter_min_stat = AMG_init_stat("/Home/damage/radszuwe/Daten/Alter_min");
	
	fclose(Output_global);
	fclose(Output_local);
}

void clean(){
	if (Elastic!=NULL) free_sparse3D(Elastic);
	if (Regularization!=NULL) free_sparse(Regularization);
	if (Interface!=NULL) free_sparse(Interface);
	if (Dissipation!=NULL) free(Dissipation);	
	if (Bound_Dirichlet!=NULL) free_sparse3D(Bound_Dirichlet);
	if (Bound_Dirichlet_reg!=NULL) free_sparse(Bound_Dirichlet_reg);
	if (Bound_Neumann!=NULL) free_sparse3D(Bound_Neumann);
	if (Bound_Neumann_reg!=NULL) free_sparse(Bound_Neumann_reg); 
}


void set_initial_conditions(double* X){
	int i;
	double x,y,r;
	int d = glob_mesh.size;
	for (i=0;i<d;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		r = sqrt(x*x+y*y);
																			// only for dimension 2
		X[i] = 0;								// ux(x,y,0)
		X[i+d] = 0;								// uy(x,y,0)
		//X[i+2*d] = 1.-0.95*exp(-10*x*x);		// h(x,y,0)								// for disc
		X[i+dimension*d] = 1.;
		
	}
}

void shift(double* X,double factor,double* Lower,int n){
	int i;
	double d;
	for (i=0;i<n;i++){
		if (!R_Newton_Condition[i] && !isnan(Lower[i])) X[i] += factor*(X[i]-Lower[i]);
	}
}

void detect_lower_equal_upper(double* Lower,double* Upper,int* Cond,sparse_matrix* A,int n,double tol){
	int i;
	for (i=0;i<n;i++) if (Cond[i]!=DIRICHLET){
		if (fabs(Upper[i]-Lower[i])<tol){
			if (A!=NULL){
				reset_row(A,i);
				insert_sparse(A,1.,i,i);
			}
			Cond[i] = DIRICHLET;
			Lower[i] = NAN;
			Upper[i] = NAN;
		}
		else Cond[i] = NOCON;
	}
}

void AMG_setting_step1(amg_system_info* AMG_info){
	AMG_print_info(NO);
	AMG_set_matrix_tolerance(step1_tol);
	AMG_set_max_iter(step1_max_iter);
	AMG_set_eps(step1_eps);
	AMG_set_smoother(&AMG_SOR_smoother);
	AMG_set_SOR_relaxation_coeff(1.6);
}

void AMG_setting_step2(amg_system_info* AMG_info){
	AMG_print_info(YES);
	AMG_set_matrix_tolerance(step2_tol);
	AMG_set_max_iter(step2_max_iter);
	AMG_set_eps(step2_eps);
	AMG_set_smoother(&AMG_parallel_Jacobi_smoother);
	RN_set_AMG_data(AMG_info);
	BI_set_AMG_setup_data(AMG_info);
}

/* X- organization

	0. u x-component
	1. u y
	2. h
	3. other scalar fields 

*/

void direct_alternate_minimization(double* X,double* Globals,double dt){
	int iter,iter2,status,constrl,constru;
	double res,g,bl,bu,d;
	double* U_prev;
	int n = glob_mesh.size;	
	int time_index = round(glob_time/dt); 											// for nonequidistant time discretizazion use interpolation
	sparse_matrix* A_step1 = NULL;
	sparse_matrix* A_step2 = NULL;
	double* H = zero_vector(n);
	double* H_prev = zero_vector(n);
	double* U = zero_vector(dimension*n);
	double* b_step1 = zero_vector(dimension*n);
	double* Kernel_step2 = generate_vector(n,1.);
	int* BC_list = get_U_condition_list();
	
	// compute initial matrices and vectors 
	copy_vector_content(X,U,0,0,dimension*n);
	copy_vector_content(X,H,dimension*n,0,n);
	A_step1 = sparse3D_vB(Elastic,H,n*dimension);
	sparse_add(A_step1,Regularization,1.);
	set_dirichlet_BC(A_step1,b_step1,BC_list,&U_bound);
	A_step2 = Interface;
		
	
	// Reflective Newton sttings
	RN_set_local_solver(&multifrontal_solver); // set Multifrontal solver
	set_RN_functional(&damage_functional);
	RN_set_gradient(&damage_gradient);
	R_Newton_Condition = zero_int_list(n);
	
	// constraints
	int* Indices = get_all_indices(n);
	double* Lower = zero_vector(n);
	double* Upper = clone_vector(All_H[time_index-1],n);
	set_RN_constraints(Indices,Lower,Upper,n);
	detect_lower_equal_upper(Lower,Upper,R_Newton_Condition,Interface,n,R_Newton_Dir_tol);	// if upper==lower then use Dirichlet
	shift(H,constraint_shift,Lower,n);														// adjust H, such that it is not located at the constraints

//scalar_sparse_mult(1000.,Interface);
//for (iter2=0;iter2<2;iter2++){
//	scalar_sparse_mult(0.1,Interface);

	iter = 0;
	do{
		// first step
		status = multifrontal_solver(A_step1,U,b_step1);	// solve elastic eq.
		if (status==FAILED){
			printf("use AMG...\n");
			amg_system_info* Info_step1 = AMG_setup(A_step1,dimension,step1_AMG_depth,
				AMG_smoothing_iter,AMG_smoothing_iter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);	
			AMG_setting_step1(Info_step1);					
			AMG_solve(Info_step1,b_step1,U,&step1_AMG_stat);
			AMG_free_data(Info_step1);			
		}
		
		if (Alt_info==YES) printf("residuum elastic part: %e\n",matrix_residuum(A_step1,U,b_step1));
		if (Strain_energy!=NULL) free(Strain_energy);
		Strain_energy = sparse3D_bilinear(Elastic,U,U);		// without Regularization
		
		// second step
		copy_vector_to(H,H_prev,n);
		set_RN_Hesse(A_step2);								// Reflective Newton settings		
		RN_set_trust_region(R_Newton_Delta);
		RN_set_kernel(Kernel_step2,n);
		reflective_newton(H);								// Reflecive Newton solve
		
		// update A1,b1		
		free_sparse(A_step1);
		free(b_step1);
		A_step1 = sparse3D_vB(Elastic,H,n*dimension);
		sparse_add(A_step1,Regularization,1.);
		b_step1 = zero_vector(dimension*n);
		set_dirichlet_BC(A_step1,b_step1,BC_list,&U_bound);
		
		
		
		res = vec_dist(H,H_prev,n)/sqrt((double)n);
		if (Alt_info==YES) printf("Alternate Minimization residuum: %e\n",res);
		iter++;
	}while(res>Alt_eps && iter<Alt_max_iter);
	AMG_add_stat(&alter_min_stat,iter,fabs(res));

//if (iter>50) printf("many iterations at level %d\n",iter2);
//}
	free(H_prev);
	
	
	// save tension (preliminary)
	int i;
	char Name[512];	
	double** T = compute_stress_tensor(U,H);
	static int counter = 0;
	sprintf(Name,"%s/%s/tension%d",Output_dir,Output_name,counter);
	print_scalar_data(Name,T[3],n);
	sprintf(Name,"%s/%s/damage%d",Output_dir,Output_name,counter);
	print_scalar_data(Name,H,n);
	sprintf(Name,"%s/%s/displacement%d",Output_dir,Output_name,counter);
	print_scalar_data(Name,U,dimension*n);
		
	for (i=0;i<4;i++) free(T[i]);
	free(T);
	counter++;
	
	//energy estimation decreasing test, maybe there is a summation error
	U_prev = All_U[time_index-1];
	H_prev = All_H[time_index-1];
	d = Damage_dissipation(H_prev,H);
	g = Gibbs_energy(U,H);
	bl = Boundary_work(U_prev,U,H);
	bu = Boundary_work(U_prev,U,H_prev);
	energy_constr_lower[time_index] = energy_constr_lower[time_index-1]+bl;
	energy_constr_upper[time_index] = energy_constr_upper[time_index-1]+bu;
	energy_dissipation[time_index] = energy_dissipation[time_index-1]+d;
	constrl = (energy_dissipation[time_index]+g-gibbs0>=energy_constr_lower[time_index]);
	constru = (energy_dissipation[time_index]+g-gibbs0<=energy_constr_upper[time_index]);
	if (!backtracking || (backtracking && constrl && constru)){
		copy_vector_content(U,X,0,0,dimension*n);
		glob_time += 2.*dt;
	}
	if (Alt_info==YES && backtracking && (!constrl || !constru)){
		printf("\nenergy condition violated: step back to time %f\n\n",glob_time-dt);
	}
	
	// always update damage field
	copy_vector_content(H,X,0,dimension*n,n);
	glob_time -= 2.*dt; 								// is compensated if bounds fulfilled
	
	// store integrals	
	Globals[0] = g-gibbs0;								//0: G(t)-G(0)
	Globals[1] = energy_dissipation[time_index];		//1: Var(H)
	Globals[2] = energy_constr_lower[time_index];		//2: work bound lower
	//Globals[2] = bl;
	Globals[3] = energy_constr_upper[time_index];		//3: work bound upper
	//Globals[4] = bu;
	Globals[4] = Interface_energy(H);					//4: interface energy
	
	// clean
	free(H);
	free(U);
	free(b_step1);
	free(Upper);
	free(Lower);
	free(Indices);
	free(Kernel_step2);
	free(BC_list);
	free(R_Newton_Condition);
}

void newton_minimization(double* X,double* Globals,double dt){	
	int i,iter,status,constrl,constru;
	double res,g,bl,bu,d;
	double* U_prev;
	double* H_prev;
	
	// compute initial matrices and vectors 
	int n = glob_mesh.size;	
	int N = (dimension+1)*n;
	int time_index = round(glob_time/dt); 											// for nonequidistant time discretizazion use interpolation
	int* BC_U = get_U_condition_list();
	double* H = &(X[dimension*n]);
	double* U = &(X[0]);
		
	// Reflective Newton settings
	RN_set_local_solver(&multifrontal_solver); 										// set Multifrontal solver
	set_RN_functional(&full_functional);
	RN_set_gradient(&full_gradient);
	RN_set_Hesse_function(&full_Hesse,N);
	R_Newton_Condition = zero_int_list(N); 										 	// init with no conditions
	double* Kernel = zero_vector(N);
	for (i=0;i<n;i++){
		if (BC_U[i]==DIRICHLET){
			R_Newton_Condition[i] = DIRICHLET;
			R_Newton_Condition[i+n] = DIRICHLET;
		}
		Kernel[i] = 0;
		Kernel[i+n] = 0;
		Kernel[i+dimension*n] = 1.;
	}			
	RN_set_kernel(Kernel,N);			
	RN_set_trust_region(R_Newton_Delta);
	
	
	// constraints
	int* Indices = get_all_indices(N);
	double* Lower = zero_vector(N);
	double* Upper = zero_vector(N);
	for (i=0;i<N;i++){
		if (i<dimension*n){
			Lower[i] = NAN;					
			Upper[i] = NAN;					
		}
		else{
			Lower[i] = 0;
			Upper[i] = All_H[time_index-1][i-dimension*n];
		}		
	}
	set_RN_constraints(Indices,Lower,Upper,N);
	detect_lower_equal_upper(Lower,Upper,R_Newton_Condition,NULL,N,R_Newton_Dir_tol);	// if upper==lower then use Dirichlet
	shift(X,constraint_shift,Lower,N);													// adjust H, such that it is not located at the constraints
	
	// solve 
	reflective_newton(X);	
	
	//print_scalar_data("/Home/damage/radszuwe/Daten/scalardata",&(X[dimension*n]),n);
	//print_vector(&(X[0]),dimension*n);	
	
		
	if (Strain_energy!=NULL) free(Strain_energy);
	Strain_energy = sparse3D_bilinear(Elastic,U,U);	
		
	//energy estimation decreasing test	
	U_prev = All_U[time_index-1];
	H_prev = All_H[time_index-1];
	d = Damage_dissipation(H_prev,H);
	g = Gibbs_energy(U,H);
	bl = Boundary_work(U_prev,U,H);
	bu = Boundary_work(U_prev,U,H_prev);
	energy_constr_lower[time_index] = energy_constr_lower[time_index-1]+bl;
	energy_constr_upper[time_index] = energy_constr_upper[time_index-1]+bu;
	energy_dissipation[time_index] = energy_dissipation[time_index-1]+d;
	constrl = (energy_dissipation[time_index]+g-gibbs0>=energy_constr_lower[time_index]);
	constru = (energy_dissipation[time_index]+g-gibbs0<=energy_constr_upper[time_index]);
	
	if (backtracking && (!constrl || !constru)){
		copy_vector_content(U_prev,X,0,0,dimension*n); 								 // use old U if not fulfilled
		glob_time -= 2.*dt;
		if (Alt_info==YES) printf("\nenergy condition violated: step back to time %f\n\n",glob_time-dt);
	}
	
	// store integrals	
	Globals[0] = g-gibbs0;								//0: G(t)-G(0)
	Globals[1] = energy_dissipation[time_index];		//1: Var(H)
	Globals[2] = energy_constr_lower[time_index];		//2: work bound lower
	Globals[3] = energy_constr_upper[time_index];		//3: work bound upper
	Globals[4] = Interface_energy(H);					//4: interface energy
	
	// clean
	free(Upper);
	free(Lower);
	free(Indices);
	free(Kernel);
	free(BC_U);
	free(R_Newton_Condition);
}

void iterative_alternate_minimization(double* X,double* Globals,double dt){
	int iter,constrl,constru;
	double res,d,g,bl,bu;
	double* U_prev;
	int n = glob_mesh.size;
	int time_index = round(glob_time/dt); 		
	amg_system_info* Info_step1 = NULL;
	amg_system_info* Info_step2 = NULL;
	sparse_matrix* A_step1 = NULL;
	sparse_matrix* A_step2 = NULL;
	double* H = zero_vector(n);
	double* H_prev = zero_vector(n);
	double* U = zero_vector(dimension*n);
	double* b_step1 = zero_vector(dimension*n);
	double* Kernel_step2 = generate_vector(n,1.);
	int* BC_list = get_U_condition_list();
	
	// compute initial matrices and vectors 
	copy_vector_content(X,U,0,0,dimension*n);
	copy_vector_content(X,H,dimension*n,0,n);
	U_prev = clone_vector(U,dimension*n);
	A_step1 = sparse3D_vB(Elastic,H,n*dimension);
	sparse_add(A_step1,Regularization,1.);
	set_dirichlet_BC(A_step1,b_step1,BC_list,&U_bound);
	Info_step1 = AMG_setup(A_step1,dimension,step1_AMG_depth,AMG_smoothing_iter,AMG_smoothing_iter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);	
	A_step2 = Interface;
	
	// BiCGstab settings	
	RN_set_solver(RN_BICGSTAB);			
	BI_set_precon(&BI_AMG_precon);
	BI_set_mat_mult(&RN_default_mat_mult);
	BI_set_eps(BiCGStab_eps);
	BI_set_max_iter(BiCGStab_max_iter);
	set_system_matrix(A_step2);
	
	Info_step2 = AMG_setup(A_step2,1,step2_AMG_depth,AMG_smoothing_iter,AMG_smoothing_iter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);	
	
	// constraints
	R_Newton_Condition = zero_int_list(n);
	int* Indices = get_all_indices(n);
	double* Lower = zero_vector(n);
	double* Upper = clone_vector(All_H[time_index-1],n);
	set_RN_constraints(Indices,Lower,Upper,n);
	detect_lower_equal_upper(Lower,Upper,R_Newton_Condition,Interface,n,R_Newton_Dir_tol);	// if upper==lower then use Dirichlet
	shift(H,constraint_shift,Lower,n);						// adjust H, such that it is not located at the constraints

	iter = 0;
	do{
		// first step
		AMG_setting_step1(Info_step1);						// main AMG solver settings	
		AMG_solve(Info_step1,b_step1,U,&step1_AMG_stat);	// AMG solve
		if (Strain_energy!=NULL) free(Strain_energy);
		Strain_energy = sparse3D_bilinear(Elastic,U,U);
		
		// second step
		copy_vector_to(H,H_prev,n);
		set_RN_Hesse(A_step2);								// Reflective Newton settings		
		RN_set_trust_region(R_Newton_Delta);
		RN_set_kernel(Kernel_step2,n);
		AMG_setting_step2(Info_step2);						// preconditioner settings for BiCGStab
		reflective_newton(H);								// Reflecive Newton solve
		
		// update A1,b1		
		free_sparse(A_step1);
		free(b_step1);
		A_step1 = sparse3D_vB(Elastic,H,n*dimension);
		b_step1 = zero_vector(dimension*n);
		set_dirichlet_BC(A_step1,b_step1,BC_list,&U_bound);
		AMG_recompute_coarse_matrices(Info_step1,A_step1);
		
		double* Z = clone_vector(H,n);
		//vector_add(Z,&(X[2*n]),-1.,n);
		print_scalar_data("/Home/damage/radszuwe/Daten/scalardata",Z,n);
		print_vector(U,2*n);
		free(Z);
		
		res = vec_dist(H,H_prev,n)/sqrt((double)n);
		if (Alt_info==YES) printf("Alternate Minimization residuum: %e\n",res);
		iter++;
	}while(res>Alt_eps && iter<Alt_max_iter);
	AMG_add_stat(&alter_min_stat,iter,fabs(res));
	free(H_prev);
	
	//energy estimation decreasing test
	U_prev = All_U[time_index-1];
	H_prev = All_H[time_index-1];
	d = Damage_dissipation(H_prev,H);
	g = Gibbs_energy(U,H);
	bl = Boundary_work(U_prev,U,H);
	bu = Boundary_work(U_prev,U,H_prev);
	energy_constr_lower[time_index] = energy_constr_lower[time_index-1]+bl;
	energy_constr_upper[time_index] = energy_constr_upper[time_index-1]+bu;
	energy_dissipation[time_index] = energy_dissipation[time_index-1]+d;
	constrl = (energy_dissipation[time_index]+g-gibbs0>=energy_constr_lower[time_index]);
	constru = (energy_dissipation[time_index]+g-gibbs0<=energy_constr_upper[time_index]);
	if (!backtracking || (backtracking && constrl && constru)){
		copy_vector_content(U,X,0,0,dimension*n);
		glob_time += 2.*dt;
	}
	if  (backtracking && (!constrl || !constru)){
		printf("\nenergy condition violated: step back to time %f\n\n",glob_time-dt);
	}
	
	// always update damage field
	copy_vector_content(H,X,0,dimension*n,n);
	glob_time -= 2.*dt; 								// is compensated if bounds fulfilled
	
	// store integrals	
	Globals[0] = g-gibbs0;								//0: G(t)-G(0)
	Globals[1] = energy_dissipation[time_index];		//1: Var(H)
	Globals[2] = energy_constr_lower[time_index];		//2: work bound lower
	Globals[3] = energy_constr_upper[time_index];		//3: work bound upper
	Globals[4] = Interface_energy(H);					//4: interface energy
	
	// clean
	AMG_free_data(Info_step1);free(Info_step1);
	AMG_free_data(Info_step2);free(Info_step2);
	free(H);
	free(U);
	free(U_prev);
	free(b_step1);
	free(Upper);
	free(Lower);
	free(Indices);
	free(Kernel_step2);
	free(BC_list);
	free(R_Newton_Condition);
}

void solve(double* X,double* Globals,double time,int steps){
	int i,j,n;
	double* Prev;
	int total_maxiter; 
	
	// main loop settings
	glob_time = 0;
	time_steps = steps;
	total_maxiter = 10*time_steps;
	dt = (double)time/steps;
	n = glob_mesh.size;
	
	// Reflective Newton settings;
	R_Newton_Delta = sqrt((double)n)/2.;
	R_Newton_trust_radius_max = 2.*R_Newton_Delta;
	RN_set_params(R_Newton_max_iter,R_Newton_eps,R_Newton_sigma_l,R_Newton_sigma_u,R_Newton_rho,R_Newton_Delta,
	R_Newton_Xi,R_Newton_trust_lower,R_Newton_trust_upper,R_Newton_trust_radius_min,R_Newton_trust_radius_max);
	
	// set initial conditions
	All_U = (double**)malloc((steps+1)*sizeof(double*));
	All_H = (double**)malloc((steps+1)*sizeof(double*));
	All_U[0] = clone_vector(X,dimension*n);									// if U is at index 0
	All_H[0] = clone_vector(&(X[dimension*n]),n);				
	for (i=1;i<=steps;i++){
		All_U[i] = zero_vector(dimension*n);
		All_H[i] = zero_vector(n);
	}
	energy_dissipation = zero_vector(steps+1);
	energy_constr_lower = zero_vector(steps+1);
	energy_constr_upper = zero_vector(steps+1);
	energy_constr_lower[0] = -energy_tol;
	energy_constr_upper[0] = +energy_tol;
	gibbs0 = Gibbs_energy(X,&(X[dimension*n]));
	
	// save initial
	FILE* Output_global = open_global();
	fprintf(Output_global,"%d\t%d\n",steps,global_size);
	fflush(Output_global);
	fclose(Output_global);
	//save(X,NULL,global_size,glob_time);
	
	printf("\nstart computation ...\n");
	i = 0;
	do{
		i++;
		glob_time += dt;
		j = round(glob_time/dt);
		direct_alternate_minimization(X,Globals,dt);
		//newton_minimization(X,Globals,dt);
		copy_vector_content(X,All_U[j],0,0,dimension*n);
		copy_vector_content(X,All_H[j],dimension*n,0,n);
		
		if (j % sample_interval == 0){
			save(X,Globals,dimension+1,global_size,glob_time);
		}
		printf("\rtime: %f max: %f",glob_time,time);
		fflush(stdout);
		
		/*p = 100*i/steps;
		printf("\rprogress: %d%%",p);
		fflush(stdout);*/				
	}while(glob_time>= 0 && glob_time<time && i<total_maxiter);
	
	printf("\nfinished\n\n");
	
	// clean
	for (i=0;i<=steps;i++){
		free(All_U[i]);
		free(All_H[i]);
	}
	free(All_U);
	free(All_H);
	free(energy_dissipation);
	free(energy_constr_lower);
	free(energy_constr_upper);
}


/*double* get_reference_initial(double L,int n){
	int i;
	double eps = 0.1;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++) Res[i] = (double)i/(n-1)*L+eps;
	return Res;
}*/

void reference_disc(double* Lower,double* Upper,point2D* Pos,point2D* Shift){
	int i;
	int n = glob_mesh.size;
	int m = get_center_index();
	Shift->x = Pos->x;
	Shift->y = Pos->y;
	vec_add_mult(Shift,&(glob_mesh.Points[m]),-1.);
	for (i=0;i<n;i++){
		if (!isnan(Lower[i])) Lower[i] -= glob_mesh.Points[i].x+Shift->x;
		if (!isnan(Lower[i+n])) Lower[i+n] -= glob_mesh.Points[i].y+Shift->y;
		
		if (!isnan(Upper[i])) Upper[i] -= glob_mesh.Points[i].x+Shift->x; 
		if (!isnan(Upper[i+n])) Upper[i+n] -= glob_mesh.Points[i].y+Shift->y;
	}
}

/*double* get_energy(sparse_matrix* A,double* X){
	int i;
	int n = glob_mesh.size;
	double* Res = sparse_mult(A,X);
	for (i=0;i<n;i++){
		Res[i] += Res[i+n];
		Res[i] /= 2.*vector_Ui_0(i);
	}
	return Res;
}*/


void smoother(sparse_matrix* A,double* F,double* Sol,int deg_freedom,int iter){
	int i,j,k,l,I,ind,diag,bottom,top,d,mesh_ind;							
	const int chunk_size = 5000;
	double sum;
	int n = A->size;
	int N = n/deg_freedom;
	double* Old = zero_vector(n);
	
	
	for (k=0;k<iter;k++){		
		copy_vector_to(Sol,Old,n);
#line 1599 "crack.c"
		#pragma omp parallel private(i,j,d,I,sum,diag,ind)
		{
#line 1601 "crack.c"
			#pragma omp for schedule(dynamic,chunk_size) 
			for (i=0;i<N;i++){
				for (d=0;d<deg_freedom;d++){
					I = i+N*d;
					
						sum = F[I];
						diag = -1;
						for (j=0;j<A->Len[I];j++){
							ind = A->Indices[I][j];
							if (ind!=I) sum -= A->Values[I][j]*Old[ind];
							else diag = j;
						}
						//#pragma omp critical 
						Sol[I] = sum/A->Values[I][diag];
										
				}
				//Old[i] += (double)i/N;				
			}
		}
	}
	free(Old);		
}
















///////////////////////////// test functions /////////////////////////////////////////

double Fx(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	return cos(4.*x)-1.-y*y;		
}
	
double Fy(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	return -4.*y*y;		
}

double f(double x){
	return -(x-1)*(x-1)+(x-1)*(x-1)*(x-1)*(x-1);
}

void elastic_energy_1D(sparse_matrix* A,double* b,double L){
	int i;
	int n = A->size;
	double h = (double)L/(n-1);
	
	insert_sparse(A,1./h,0,0);
	insert_sparse(A,-1./h,0,1);
	insert_sparse(A,1./h,n-1,n-1);
	insert_sparse(A,-1./h,n-1,n-2);
	b[0] = h/2.;
	b[n-1] = h/2.;
	
	for (i=1;i<n-1;i++){
		insert_sparse(A,2./h,i,i);
		insert_sparse(A,-1./h,i,i-1);
		insert_sparse(A,-1./h,i,i+1);
		b[i] = h;
	}
}

void test_product(){
	
	int num_fields = 2;
	set_var_number(num_fields);	
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV_a);
	sparse_matrix* B = set_matrix_Aij_11_2D(0,0,&insert_A_abV_b);
	sparse_matrix* C = sparse_product(A,B);
	sparse_matrix* FC;
	//printf("B symmetric ? %d\n")
	product_pattern pattern = set_product_pattern(A,B,NULL,&FC);
	
	int i;
	int N = 10;
	set_thread_num(8);
	
	printf("start...\n");
	fflush(stdout);
	time_t start,fin;
	time(&start);
	
	
	
	for (i=0;i<N;i++){
		fast_sparse_product(FC,A,B,&pattern);
		//sparse_matrix* FC = sparse_product(A,B);	
	}
	
	time(&fin);
	printf("physical execution time: %d sec\n",(int)(fin-start));
	
	
	exit(0);
}

void test_BiCGStab(){
	
	int i;
	int num_fields = 2;
	set_var_number(num_fields);	
	set_thread_num(1);
	set_omp_chunk_size(100);
	
	//sparse_matrix* A = read_sparse_textfile("/Home/damage/radszuwe/Daten","matrix");
	int n = glob_mesh.size;
	set_var_number2D(2);
	sparse_matrix* A = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_abV_b);
	scalar_sparse_mult(lambda,A);
	sparse_matrix* A2 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_baV_b);
	sparse_matrix* A3 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_bbV_a);
	sparse_add(A,A2,mu);
	sparse_add(A,A3,mu);
	free_sparse(A2);
	free_sparse(A3);
		
	sparse_matrix* B3 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_abV_b);
	scalar_sparse_mult(lambda,B3);
	sparse_matrix* B1 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_baV_b);	
	sparse_matrix* B2 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_bbV_a);
	sparse_add(B3,B1,mu);
	sparse_add(B3,B2,mu);
	free_sparse(B1);
	free_sparse(B2);
	sparse_matrix* B = get_transpose(B3,2*n);
		
	sparse_matrix* Tx;
	sparse_matrix* Ty;
	sparse_matrix* Trow_x = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_x);
	sparse_matrix* Trow_y = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_y);
	scalar_sparse_mult(lambda,Trow_x);
	scalar_sparse_mult(lambda,Trow_y);
	
	Tx = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_x);
	Ty = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_y);
	sparse_add(Trow_x,Tx,mu);
	sparse_add(Trow_y,Ty,mu);
	free_sparse(Tx);
	free_sparse(Ty);
	
	Tx = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_x);
	Ty = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_y);
	sparse_add(Trow_x,Tx,mu);
	sparse_add(Trow_y,Ty,mu);
	free_sparse(Tx);
	free_sparse(Ty);
	
	double* b = zero_vector(2*n);
	glob_time = 1.;
	int* BC_list = get_U_condition_list();
	set_dirichlet_BC(A,b,BC_list,&U_bound);
	
	set_system_matrix(A);
	BI_set_mat_mult(&matvec);
	BI_set_AMG_preconditioner(A,2,3);
	
	time_t start,fin;
	time(&start);
	
	double* X = zero_vector(2*n);
	printf("start solver...\n");
	fflush(stdout);	
	i = BiCGStab_solve(X,b,2*n);
	
	printf("iterations: %d\n",i);
	
	time(&fin);
	printf("physical execution time: %d sec\n",(int)(fin-start));
	fflush(stdout);
	
	double eps = 1e-10;
	double* Fx = sparse_mult(Trow_x,X);
	double* Fy = sparse_mult(Trow_y,X);
	double* Txx = compute_scalar_function(&(Fx[0]),YES,eps);
	double* Txy = compute_scalar_function(&(Fx[n]),YES,eps);
	double* Tyx = compute_scalar_function(&(Fy[0]),YES,eps);
	double* Tyy = compute_scalar_function(&(Fy[n]),YES,eps);
	
	double* dB = sparse_mult(B,X);	
	
	print_scalar_data("/Home/damage/radszuwe/Daten/scalardata",&(dB[n]),n);
	print_vector(X,2*n);
	
	exit(0);
	
}

void test_AMG(){
	
	int i,depth,smiter;
	int num_fields = 1;
	set_var_number(num_fields);	
	
	sparse_matrix* AMG_sparse = read_sparse_textfile("/Home/damage/radszuwe/Daten","ugly_matrix");
	//sparse_matrix* AMG_sparse = set_matrix_Aij_11_2D(0,0,&insert_A_bbV_a);
	//sparse_matrix* Tr = set_matrix_Aij_11_2D(0,0,&insert_A_abV_b);
	//sparse_add(AMG_sparse,Tr,0.3);
	int n = AMG_sparse->size;
	
	
	
	
	int s = 20;
	/*sparse_matrix* H = NULL;
	double** U = (double**)malloc(s*sizeof(double*));
	for (i=0;i<s;i++) U[i] = NULL;
	double* V0 = generate_vector(n,1.);
	reortho_Arnoldi(AMG_sparse,&H,U,V0,&s,1);
	
	double* Eig = zero_vector(s);
	double** V = (double**)malloc(s*sizeof(double*));
	for (i=0;i<s;i++) V[i] = NULL;
	Ritz_values(H,V,Eig,200);*/
	
	
	double** W = (double**)malloc(s*sizeof(double*));
	/*for (i=0;i<s;i++){
		int j;
		W[i] = zero_vector(n);
		for (j=0;j<s;j++) vector_add(W[i],U[j],V[i][j],n);
		vector_normalize(W[i],n);
	}*/
	
	double* Eig = zero_vector(s);
	double* W0 = generate_vector(n,1.);
	Harmonic_Ritz_values(AMG_sparse,W,W0,Eig,s);
	
	sparse_matrix* ID = sparse_identity(n);
	for (i=0;i<s;i++){
		sparse_matrix* D = clone(AMG_sparse);
		sparse_add(D,ID,-Eig[i]);
		double* Z = sparse_mult(D,W[i]);
		printf("eigen value: %e\n",Eig[i]);
		printf("eigen residuum %d: %e\n",i,euklid_norm(Z,n)/matrix_norm(D));
		free_sparse(D);
		free(Z);
	}
	
	double eig= 1.;
	double* Vec = zero_vector(AMG_sparse->size);
	i = get_most_negative_eigenvector(AMG_sparse,&eig,Vec,10000,1e-10);
	printf("iterations: %d, smallest eigenvalue: %e\n",i,eig);
	
	vector_add(Vec,W[0],-1.,n);
	printf("dist to smallest EV: %e\n",euklid_norm(Vec,n));
	
	/*exit(0);
	free(W[s-1]);
	W[s-1] = clone_vector(Vec,n);
	Eig[s-1] = eig;*/
	
	sparse_matrix* QT = sparse_zero(s);
	convert_array_to_sparse(W,QT,s,n);
	sparse_matrix* Q = get_transpose(QT,n);
	sparse_matrix* AQ = sparse_product(AMG_sparse,Q);
	sparse_matrix* QTAQ = sparse_product(QT,AQ);
	print_sparse(QTAQ);	
	sparse_matrix* LA = sparse_zero(s);
	sparse_matrix* UA = sparse_zero(s);
	complete_LU_factorization(QTAQ,LA,UA);
	
	
	
	//index2D ind = get_most_distinct_points();
	//int Constr[4] = {ind.i,ind.i+glob_mesh.size,ind.j,ind.j+glob_mesh.size};
	
	int** AMG_cond = set_pure_conditions(num_fields,DIRICHLET);
	//sparse_matrix* AMG_Bound = FEM_get_dirichlet_correction(AMG_sparse,AMG_cond,1.);
	double* b = get_function(NULL,glob_mesh.size,num_fields,0,&Fx);
	//double* by = get_function(NULL,glob_mesh.size,num_fields,1,&Fy);			only when 2D
	//vector_add(b,by,1.,n);
	double* B = set_vector_Ui_0_2D(num_fields);
	double* Bound_val = zero_vector(n);
	vector_pseudo_mult(b,B,n);
	//linear_map(b,1.,AMG_Bound,Bound_val);
	//include_aux_constraints(AMG_sparse,b,Constr,4);
	
	AMG_set_max_iter(5);
	AMG_print_info(YES);
	AMG_set_SOR_relaxation_coeff(1.6);
	smiter = 2;
	depth = 4;
	
	amg_system_info* Setup_data = AMG_setup(AMG_sparse,num_fields,depth,smiter,smiter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);
	double* X = zero_vector(n);
	
	
	
	double* Y = zero_vector(n);

	clock_t start = clock();
	
	multifrontal_solver(AMG_sparse,Y,b);
	
	clock_t end = clock();
	
	printf("umfpack residuum: %e\n",matrix_residuum(AMG_sparse,Y,b));
	printf("\ntime: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	
	start = clock();
	
	double r0 = matrix_residuum(AMG_sparse,X,b);
	AMG_solve(Setup_data,b,X,NULL);
	
	for (i=0;i<10;i++){	
		double* R = clone_vector(b,n);
		linear_map(R,-1.,AMG_sparse,X);
		double* Rc = sparse_mult(QT,R);
		L_triang_invert(LA,Rc);
		U_triang_invert(UA,Rc);
		double* Dx = sparse_mult(Q,Rc);
		vector_add(X,Dx,1.,n);
		free(R);
		free(Rc);
		free(Dx);
		printf("norm X: %e\n",euklid_norm(X,n));
		AMG_solve(Setup_data,b,X,NULL);
	}
	
	
	double r = matrix_residuum(AMG_sparse,X,b);
	
	end = clock();
	printf("\ntime: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	printf("overall residuum: %e\n",r/r0);
	
	//double* Shift = clone_vector(X,n);//zero_vector(2*n);
	//print_vector(Shift,n);
	//print_scalar_data("/Home/damage/radszuwe/Daten/amg",X,n);
	//print_vector(sparse_row_sum(Setup_data->Prolongation[1]),(Setup_data->Prolongation[1])->size);
	exit(0);
}

void test_1D(int n){
	int i;
	double x,eig;
	double* F = zero_vector(n);
	
	for (i=0;i<n;i++){
		x = (double)i/(n-1);
		F[i] = 0.*exp(-10.0*(x-0.5)*(x-0.5));
	}
	
	//sparse_matrix* A = get_1D_Laplace(F,n);
	sparse_matrix* A = get_test_matrix(n,0);
	double* V1 = zero_vector(n);
	double* V2 = zero_vector(n);
	double* X = zero_vector(n);
	V1[0] = 1.0/sqrt(2.0);
	V1[3] = 1.0/sqrt(2.0);
	V2[1] = 1.0;
	F[1] = 0.014;
	//F[0] = -0.02;
	subspace_minimum(A,F,V1,V2,X);
	print_sparse(A);
	print_vector(X,n);
	exit(0);
	
	double* V = generate_vector(n,1./sqrt((double)n));
	printf("start computation\n");
	int iter = get_most_negative_eigenvector(A,&eig,V,100000,1e-8);
	printf("itrations: %d\n",iter);
	printf("minimal eigenvalue: %f\n",eig);
	
	free_sparse(A);
	free(F);
}

/*int main(int argc, char* argv[]){
		
		char* Dir = "/Home/damage/radszuwe/Daten";	
		char* Name = argv[1];
		read_mesh_2D(Dir,Name);	
			
		
		int n = glob_mesh.size;
		set_var_number2D(2);
		set_thread_num(2);
		
		//test_AMG();
		//test_product();
		//test_BiCGStab();
		
		double mu = 1.;
		double lambda = -1.;
		sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV_a);
		scalar_sparse_mult(mu,A);											// factor 1/2 included autmomatically
		sparse_matrix* B = set_matrix_Aij_11_2D(0,0,&insert_A_abV_b);		
		sparse_add(A,B,(mu+lambda));
		free_sparse(B);
		
		point2D g = init_point2D(0.1,0.2);
		double* G = generate_2D_vector(&g,n);
		double* b = set_vector_bi_0(0,&insert_Ba);
		//double* G = zero_vector(2*n);
		//for (i=0;i<n;i++) G[i] = g.x;
		//for (i=n;i<2*n;i++) G[i] = g.y;
		vector_pseudo_mult(b,G,2*n);
		free(G);
		double* X = zero_vector(2*n);
		for (i=n;i<2*n;i++) X[i] += 0.05;
		
		point2D pos = init_point2D(0,0);
		point2D shift = init_point2D(0,0);
		int* Indices = get_all_indices(2*n);
		double* Lower = generate_vector(2*n,-2.);
		double* Upper = generate_vector(2*n,2.);		
		reference_disc(Lower,Upper,&pos,&shift);
		set_RN_constraints(Indices,Lower,Upper,2*n);
		
		// reduce bandwidth
		
		int* Map = zero_int_list(2*n);
		int* Inv_Map = zero_int_list(2*n);
		int w0 = get_ave_band_width(A);
		sparse_matrix* PA = sparse_zero(2*n);
		sparse_get_optimized_index_map(A,Map,Inv_Map);
		map_indices(A,PA,NULL,NULL,Map);
		int w = get_ave_band_width(PA);		
		print_sparse(PA);
		printf("bandwidth reduction: %f\n",(double)w/w0);
		
		vector_permutation(X,Inv_Map,2*n);
		vector_permutation(b,Inv_Map,2*n);
		vector_permutation(Lower,Inv_Map,2*n);
		vector_permutation(Upper,Inv_Map,2*n);
		
		set_RN_Hesse(PA);		// when bandwidth reduced PA else A
		set_RN_linear(b);
		set_RN_functional(NULL);
		
		
		
		
		
		#ifdef __CUDACC__			
			double* Y = clone_vector(X,2*n);
			int iterations = 30;
			cuda_gpu_settings(2);
			printf("\n cuda: start ...\n");
			fflush(stdout);
		
			cudaEvent_t start,stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
				
		
			
			float Time;
			
			
			//int* _Ind = NULL;
			//double* _Val = NULL;
			//int width = get_max_mem_bandwidth(A);
			//cuda_sparse_on_device(A,width,&_Ind,&_Val);
			
			csr_matrix _s;
			float* _X = cuda_zero_vector(2*n);
			float* _b = cuda_get_vector_on_device(b,2*n);
			sparse_to_csr_device(&_s,A,STORE_INV_DIAG);		
			//device_print_vector(_s.Val,_s.total_size);
			//print_sparse(A);
			
			cudaEventRecord(start,0);
			cuda_csr_jacobi(&_s,_X,_b,iterations);
				
			//cudaFree(_Ind);
			//cudaFree(_Val);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
					
			cudaEventElapsedTime( &Time, start, stop );
			
			printf("physical execution time: %f sec\n",Time/1000.);
	
			
			
	
			
			printf("\n default: start ...\n");

			
			cudaEventRecord(start,0);
			AMG_Jacobi_smoother(A,b,Y,NULL,0,0,iterations);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime( &Time, start, stop );
			
			printf("physical execution time: %f sec\n",Time/1000.);
			fflush(stdout);
			
			vector_add(X,Y,-1,2*n);
			printf("difference norm: %e\n",euklid_norm(X,2*n));
			
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			exit(0);
		#endif
		
		int iter = reflective_newton(X);
		double* Shift = generate_2D_vector(&shift,n);
		
		vector_permutation(X,Map,2*n);  // when bandwidth reduced
		
		vector_add(X,Shift,1.,2*n);
		print_vector(X,2*n);
		printf("finished !\nReflective Newton iterations: %d\n\n",iter);
		print_RN_statistics(stdout);
				
		double* data = compute_divergence(X);
		sprintf(Name,"%s/scalardata",Dir);
		print_scalar_data(Name,data,n);		
		
		free_sparse(A);
		free(Lower);
		free(Upper);
		free(Indices);
		free(Shift);
		free(b);
		free(X);
		free(data);
}*/













void test_arg(int argc,int n){
	if (argc<n){
		printf("too few arguments -> abort\n");
		exit(0);
	}
}

///////////////////////////////////////////// main program //////////////////////////////////////////

int main(int argc, char* argv[]){
	
	int i;
	test_arg(argc,2);
	
	if (strcmp(argv[1],"sim")==0){
		time_t start,fin;
		test_arg(argc,4);
		
		// here: read all parameters
		sprintf(Mesh_dir,"%s/%s",getenv("HOME"),"Daten");
		sprintf(Mesh_name,"%s",argv[2]);
		
		sprintf(Output_dir,"%s/%s",getenv("HOME"),"Daten");
		sprintf(Output_name,"%s",argv[3]);
		
		setup();		
		
		//test_BiCGStab();
		
		double* Globals = zero_vector(global_size);
		double* Solution = zero_vector(glob_mesh.size*(dimension+1));
		set_initial_conditions(Solution);
		
		time_steps = 100;
		total_time = 1.;
		
		time(&start);
		
		// main solve
		solve(Solution,Globals,total_time,time_steps);
		
		time(&fin);
		printf("physical execution time: %d sec\n",(int)(fin-start));
		fflush(stdout);
		
		//when iterative solvers are used:
		//AMG_print_stat(BI_get_AMG_stat());
		//AMG_print_stat(BI_get_stat());
		//AMG_print_stat(&step1_AMG_stat);
		
		print_RN_statistics(stdout);
		AMG_print_stat(&alter_min_stat);
		
		//print_scalar_data("/Home/damage/radszuwe/Daten/scalardata",&(Solution[dimension*glob_mesh.size]),glob_mesh.size);
		//print_vector(&(Solution[0]),dimension*glob_mesh.size);	
		
		free(Solution);
		free(Globals);
		clean();
	}
	if (strcmp(argv[1],"mesh")==0){	
		double a;
		test_arg(argc,3);
		
		// mesh creation
		char FullName[512];
		char* Dir = "/Home/damage/radszuwe/Daten";	
		char* Name = argv[2];
		
		int b_num = atoi(argv[3]);
		sprintf(FullName,"%s/%s",Dir,Name);
		printf("create mesh data in file %s\n",FullName);
		strcat(FullName,".poly");
		
		//a = create_2D_mesh(FullName,b_num);								// disc
		a = create_shape2(FullName,b_num);									// polygon
		
		char Command[512];
		sprintf(Command,"triangle -p -v -q30 -a%f %s",a,FullName);
		printf("execute: %s\n",Command);
		system(Command);
		
		if (argc==5 && strcmp(argv[4],"-refine")==0){
			char NewName[512];
			sprintf(NewName,"%s.1",Name);	
			read_mesh_2D(Dir,NewName);					
			sprintf(FullName,"%s/%s.1",Dir,Name);		
			refine_mesh(FullName,&F_area_shape2);		
		}		
	}
	return 0;
}
