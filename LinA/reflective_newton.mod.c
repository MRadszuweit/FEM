#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "reflective_newton.c.opari.inc"
#line 1 "reflective_newton.c"

#include "reflective_newton.h"

///////////////////////////////// global variables ///////////////////////////////

static int RN_max_iter = 50;
static double RN_eps = 1e-8;
static double RN_sigma_l = 0.1;
static double RN_sigma_u = 0.9;
static double RN_rho = 0.1;
static double RN_Delta = 250.;					// sqrt(n)/2
static double RN_Xi = 1.;
static double RN_trust_lower = 0.6;
static double RN_trust_upper = 1.1;
static double RN_trust_radius_min = 1e-3;
static double RN_trust_radius_max = 1000.;

static int RN_solver_type = RN_BICGSTAB;
static RN_statistics statistics;

static int RN_PCG_iter = 100;
const int eig_iter = 10000;			// maximal number of iterations used to cumpute negative curvature direction
const double eig_eps = 1e-6;		// precision to define converged state 
const double eig_neg_eps = 1e-4;	// minimum distance of negative curvature from zero 

static int RN_dim = 0;
static int* RN_bound = NULL;
static int RN_bound_num = 0;
static int RN_status = POSITIVE_DEFINITE;
static double* RN_upper = NULL;
static double* RN_lower = NULL;
static double RN_ax = 0;
static double RN_ay = 0;
static double RN_axy = 0;
static double RN_bx = 0;
static double RN_by = 0;

static int (*RN_multifrontal_solver)(sparse_matrix* A,double* X,double* b);
static double (*RN_Functional)(double* X,int n) = NULL;
static double* (*RN_gradient)(double* X) = NULL;
static sparse_matrix* (*RN_compute_Hesse)(double* X) = NULL;
static sparse_matrix* RN_Hesse = NULL;
static sparse_matrix* RN_M = NULL;
static sparse_matrix* RN_L = NULL;
static sparse_matrix* RN_U = NULL;
static double* RN_b = NULL;
static double* Kernel_dir = NULL;
static amg_system_info* RN_AMG_setup_data = NULL;

static int RN_info = RN_YES;
static int Counter = 0;

///////////////////////////////// Functions ///////////////////////////////////////

double RN_min(double a,double b){
	if (a<b) return a; else return b;
}

double RN_max(double a,double b){
	if (a>b) return a; else return b;
}

int* get_all_indices(int size){
	int i;
	int* Res = zero_int_list(size);
	for (i=0;i<size;i++) Res[i] = i;
	return Res;
}

double* RN_default_gradient(double* X){												//    nur für den Spezialfall q(x)=x.b+x.H.x/2
	double* G = clone_vector(RN_b,RN_dim);
	linear_map(G,1.,RN_Hesse,X);
	return G;
}

double RN_default_func(double* X, int n){											//    nur für den Spezialfall q(x)=x.b+x.H.x/2
	return sparse_bilinear(X,RN_Hesse,X)/2.+scalar(RN_b,X,n);
}

void RN_print_info(int mode){
	RN_info = mode;
}

void RN_set_params(
	int R_Newton_max_iter,
	double R_Newton_eps,
	double R_Newton_sigma_l,
	double R_Newton_sigma_u,
	double R_Newton_rho,
	double R_Newton_Delta,
	double R_Newton_Xi,
	double R_Newton_trust_lower,
	double R_Newton_trust_upper,
	double R_Newton_trust_radius_min,
	double R_Newton_trust_radius_max){
RN_max_iter = R_Newton_max_iter;
RN_eps = R_Newton_eps;
RN_sigma_l = R_Newton_sigma_l;
RN_sigma_u = R_Newton_sigma_u;
RN_rho = R_Newton_rho;
RN_Delta = R_Newton_Delta;
RN_Xi = R_Newton_Xi;
RN_trust_lower = R_Newton_trust_lower;
RN_trust_upper = R_Newton_trust_upper;
RN_trust_radius_min = R_Newton_trust_radius_min;
RN_trust_radius_max = R_Newton_trust_radius_max;
}

void RN_set_trust_region(double R_Newton_Delta){
	RN_Delta = R_Newton_Delta;
}

void RN_init_statistics(){
	statistics.calls = 0;
	statistics.res = 0;
	statistics.average_eig_iter = 0;
	statistics.number_neg_curv = 0;
	statistics.total_iter = 0;
	statistics.line_search_count = 0;
	statistics.line_search_iter = 0;
}

void print_RN_statistics(FILE* file){
	if (statistics.total_iter!=0){
		if (statistics.number_neg_curv!=0) statistics.average_eig_iter = statistics.average_eig_iter / statistics.number_neg_curv;
	}
	if (statistics.line_search_count!=0) statistics.line_search_iter = statistics.line_search_iter / statistics.line_search_count;
	fprintf(file,"\nReflective Newton statistics:\n");
	fprintf(file,"average number of Newton iterations: %f\n",(float)statistics.total_iter/statistics.calls);
	fprintf(file,"average residual: %e\n",(float)statistics.res/statistics.calls);
	fprintf(file,"number of neg. curvature events: %d\n",statistics.number_neg_curv);
	if (statistics.number_neg_curv!=0) printf("\tfind eigenvector: average number of power iterations: %d\n",statistics.average_eig_iter);
	fprintf(file,"line searches: %d\n",statistics.line_search_count);
	if (statistics.line_search_count!=0) fprintf(file,"\taverage line search iterations: %d\n\n",statistics.line_search_iter);
}

void RN_default_mat_mult(int n,double* y,double* z){
	int i,j;
	double sum;
	if (RN_M!=NULL){
		for (i=0;i<n;i++){
			sum = 0;
			for (j=0;j<RN_M->Len[i];j++){sum += RN_M->Values[i][j]*y[RN_M->Indices[i][j]];}
			z[i] = sum;
		}
	}
	else{
		printf("error in mat_mult: no system matrix set -> abort\n");
		exit(0);
	}
}

void RN_set_default_matrix(sparse_matrix* M){
	RN_M = M;
}

void RN_set_Hesse_function(sparse_matrix* (*Hesse_func)(double* X),int dim_Hesse){
	RN_compute_Hesse = Hesse_func;
	RN_dim = dim_Hesse;
	RN_Hesse = NULL;
}
	
void RN_set_gradient(double* (*Gradient_func)(double* X)){
	RN_gradient = Gradient_func;
} 

void RN_set_AMG_data(amg_system_info* AMG_info){
	RN_AMG_setup_data = AMG_info;
}

/*void RN_init_AMG(sparse_matrix* M){
	const int smoothing_iter = 2;
	const int deg_freedom_per_point = 2;
	const int depth = 5;
	
	AMG_print_info(RN_NO);
	AMG_set_SOR_relaxation_coeff(1.6);
	
	RN_AMG_setup_data = AMG_setup(M,deg_freedom_per_point,depth,smoothing_iter,smoothing_iter+1,STANDARD,STAND_ALONE,DIRECT,NULL,NULL);	
}*/

void RN_set_local_solver(int (*Solver)(sparse_matrix* A,double* X,double* b)){
	RN_multifrontal_solver = Solver;
	if (Solver!=NULL) RN_solver_type = RN_UMF;
}

void RN_set_solver(int type){
	RN_solver_type = type;
}

void RN_init_BiCGStab(sparse_matrix* M){
	
	const double BI_eps = 1e-3;					// residuum reduction in each BiCGStab 
	const int BI_max_iter = 10;					// maximum iteration number
	const int degree_of_freedom = 2;			// number of degrees of freedom per point (important !)
	const int depth = 6;						// AMG preconditioner depth
	
	BI_set_eps(BI_eps);
	BI_set_max_iter(BI_max_iter);
	BI_set_mat_mult(&RN_default_mat_mult);	
	BI_set_AMG_preconditioner(M,degree_of_freedom,depth);
	RN_AMG_setup_data = BI_get_AMG_setup_data();
}

void RN_set_kernel(double* K,int n){
	if (n==RN_dim) Kernel_dir = K;
	else{
		printf("kernel vector dimension does not match system dimension -> abort\n");
		exit(0);
	}
}

void RN_set_D_J(double* Gradient, double* X,double* D,double* J){
	int i,index;
	for (i=0;i<RN_dim;i++){
		D[i] = 1.;
		J[i] = 0.;
	}
	for (i=0;i<RN_bound_num;i++){
		index = RN_bound[i];
		if (Gradient[index]<0){
			if (!isnan(RN_upper[i])){
				D[index] = sqrt(fabs(X[index]-RN_upper[i]));
				J[index] = 1.;
			}
		}
		else{
			if (!isnan(RN_lower[i])){
				D[index] = sqrt(fabs(X[index]-RN_lower[i]));
				J[index] = 1.;
			}
		}
	}
}

sparse_matrix* RN_get_M_D(double* Gradient, double* X, double* D){
	int i;
	double* J = zero_vector(RN_dim);
	RN_set_D_J(Gradient,X,D,J);
	sparse_matrix* sparse_D = sparse_diagonal(D,RN_dim);
	sparse_matrix* HD = sparse_product(RN_Hesse,sparse_D);
	sparse_matrix* M = sparse_product(sparse_D,HD);
	free_sparse(HD);
	free_sparse(sparse_D);
	for (i=0;i<RN_dim;i++) if (J[i]!=0) insert_sparse(M,fabs(Gradient[i])*J[i],i,i);
	free(J);
	return M;
}

void RN_invert_D(double* D,double* X){
	int i;
	for (i=0;i<RN_bound_num;i++){
		if(D[i]!=0) X[RN_bound[i]] /= D[i];
		else{
			printf("error in RN_get_M: division by zero -> abort\n");
			exit(0);
		}
	}
}

void RN_reflect(double* X){
	int i,index;
	double w,dummy,d;
	for (i=0;i<RN_bound_num;i++){
		index = RN_bound[i];
		if (!isnan(RN_lower[i])){
			if (!isnan(RN_upper[i])){
				d = 2.*(RN_upper[i]-RN_lower[i]);
				w = modf(fabs(X[index]-RN_lower[i])/d,&dummy)*d;
				X[index] = RN_lower[i]+RN_min(w,d-w);
			}
			else X[index] = RN_lower[i]+fabs(X[index]-RN_lower[i]);			
		}
		else{
			if (X[index]>RN_upper[i]) X[index] = 2.*RN_upper[i]-X[index];
		}
	}
}

void RN_get_subspace(sparse_matrix* M,double* Gradient,double* D,double* X,double* V1,double* V2){	
	int iter;
	copy_vector_to(Gradient,V1,RN_dim);											// V1,V2 zero vectors	
	vector_pseudo_mult(V1,D,RN_dim);
	copy_vector_to(V1,V2,RN_dim);
	
	scalar_mult(-1.,V2,RN_dim);	
	double* b = clone_vector(V2,RN_dim);		
	RN_set_default_matrix(M);
	
	//// begin local solver ////////////////////////
	
	switch(RN_solver_type){
		case RN_UMF:																			// unsymmetric multifrontal solver
			RN_status = (*RN_multifrontal_solver)(M,V2,b);
			iter = 1;
			break;
		case RN_BICGSTAB:																		// BiCGStab
			AMG_recompute_coarse_matrices(RN_AMG_setup_data,M);										// for AMG preconditioner
			iter = BiCGStab_solve(V2,b,RN_dim);											
			if (iter<BI_get_max_iter()) RN_status = POSITIVE_DEFINITE; else RN_status = INDEFINITE;
			break;
		case RN_AMG:																			// AMG
			AMG_recompute_coarse_matrices(RN_AMG_setup_data,M);	
			iter = AMG_solve(RN_AMG_setup_data,b,V2,NULL);				
			if (iter<AMG_get_max_iter()) RN_status = POSITIVE_DEFINITE; else RN_status = INDEFINITE;
			break;
		case RN_PCG:
			RN_L = sparse_zero(RN_dim);
			RN_U = sparse_zero(RN_dim);
			RN_status = incomplete_LU_factorization(M,RN_L,RN_U);								// ILU-PCG
			if (RN_status==INDEFINITE) break;
			PCG_default(M,RN_L,RN_U);	
			iter = pcg(V2,b,RN_dim,RN_eps,RN_PCG_iter);		
			if (iter<RN_PCG_iter) RN_status = POSITIVE_DEFINITE; else RN_status = INDEFINITE;    
			free_sparse(RN_L);
			free_sparse(RN_U);
			break;
		case RN_CHOLESKY:																		// complete Cholesky
			iter = 1;										
			RN_L = sparse_zero(RN_dim);
			RN_U = sparse_zero(RN_dim);
			RN_status = cholesky(M,RN_L,RN_U);					
			if (RN_status==INDEFINITE) break;
			L_triang_invert(RN_L,V2);								
			U_triang_invert(RN_U,V2);
			free_sparse(RN_L);
			free_sparse(RN_U);
			break;
	}
	free(b);
	
	////// end local solver //////////////////
	
	if (RN_status==INDEFINITE){
		double curv = 0;
		statistics.number_neg_curv++;
		iter = get_neg_curvature_direction(M,&curv,V2,eig_iter,eig_eps,eig_neg_eps);		// Verbessurung: Benutze als letztes berechneten Eigenvector
		statistics.average_eig_iter += iter;
		if (curv>=0) {
			if (RN_info==YES) printf("Reflective Newton: not able to determine direction of negative curvature -> abort\n");
			if (Kernel_dir!=NULL){
				int i;
				for (i=0;i<RN_dim;i++) V2[i] = Kernel_dir[i];
				vector_normalize(V2,RN_dim);
			}			
		}
		else if (RN_info==YES){
			printf("\nnegative curvature: %f\n",curv);
			printf("found after %d iterations\n\n",iter);
		}	
	}
}

double quadratic_on_circle(double x){
	double c = cos(x)*RN_Delta;
	double s = sin(x)*RN_Delta;
	return c*(c*RN_ax+s*RN_axy+RN_bx)+s*(s*RN_ay+RN_by);
}

double RN_derivative(double x){
	return ((sin(2.*x)*(-RN_ax+RN_ay)+cos(2.*x)*RN_axy)*RN_Delta-sin(x)*RN_bx+cos(x)*RN_by)*RN_Delta;
}

void RN_print_function(double (*F)(double x),int n,double min,double max){
	int i;
	double x;
	char* Name = "/Home/damage/radszuwe/Daten/func";
	FILE* file = fopen(Name,"w");
	if (file==NULL){
		printf("could not open file %s -> abort\n",Name);
		exit(0);
	}
	for (i=0;i<n;i++){
		x = min+(max-min)*((double)i/(n-1));
		fprintf(file,"%f\t%e\n",x,(*F)(x));		
	}
	fclose(file);
}

void subspace_minimum(sparse_matrix* A,double* b,double* V1,double* V2,double* X){  //  with constraint ||X||=RN_Delta
	
	const int Newton_iter = 500;
	const double Newton_eps = 1e-6;
	
	int i,m;
	double t,d2,min_val;
	double* Roots = NULL;
	
	RN_dim = A->size;
	
	RN_bx = scalar(V1,b,RN_dim);		// q(x,y) = ax*x^2+ay*y^2+axy*x*y+bx*x+by*y
	RN_by = scalar(V2,b,RN_dim);
	RN_ax = sparse_bilinear(V1,A,V1);
	RN_ay = sparse_bilinear(V2,A,V2);
	RN_axy = sparse_bilinear(V1,A,V2)+sparse_bilinear(V2,A,V1);
	
	int root_num = Newton_1D(&RN_derivative,&Roots,0,2.*M_PI,Newton_iter,Newton_eps);
	if (root_num>0){
		m = 0;
		min_val = quadratic_on_circle(Roots[0]);	
		for (i=1;i<=root_num;i++){
			if (quadratic_on_circle(Roots[i])<min_val){
				m = i;
				min_val = quadratic_on_circle(Roots[i]);
			}
		}
		copy_vector_to(V1,X,RN_dim);
		scalar_mult(RN_Delta*cos(Roots[m]),X,RN_dim);
		vector_add(X,V2,RN_Delta*sin(Roots[m]),RN_dim);
		free(Roots);
	}
	else{
		printf("subspace_minimum: no extrema found -> abort\n");
		RN_print_function(&RN_derivative,300,0,2*M_PI);		
		exit(0);
	}
}

/*int RN_line_search_conditions(double* X_old,double* X,double* S,double* Gradient,double alpha){  // S, Grad not scaled by D ! 
	double dq = (*RN_Functional)(X,RN_dim)-(*RN_Functional)(X_old,RN_dim);
	double lower = RN_sigma_l*alpha*(scalar(Gradient,S,RN_dim)+alpha*RN_min(sparse_bilinear(S,RN_Hesse,S),0)/2.);
	//double upper = RN_sigma_u/RN_sigma_l*lower;
	//if (dq<lower && dq > upper) return 1; else return 0;		// strong condition
	if (dq<lower) return 1; else return 0;						// weaker condition
}*/

int RN_test_sigma_l(double* X_old,double* X,double* S,double* Gradient,double alpha){ 
	double dq = (*RN_Functional)(X,RN_dim)-(*RN_Functional)(X_old,RN_dim);
	double lower = RN_sigma_l*alpha*(scalar(Gradient,S,RN_dim)+alpha*RN_min(sparse_bilinear(S,RN_Hesse,S),0)/2.);
	if (dq<lower) return 1; else return 0;		
}

int RN_test_sigma_u(double* X_old,double* X,double* S,double* Gradient,double alpha){ 
	double dq = (*RN_Functional)(X,RN_dim)-(*RN_Functional)(X_old,RN_dim);
	double upper = RN_sigma_u*alpha*(scalar(Gradient,S,RN_dim)+alpha*RN_min(sparse_bilinear(S,RN_Hesse,S),0)/2.);
	if (dq>upper) return 1; else return 0;	
}

double predicted_readuction(double* S,double* Gradient){
	return scalar(Gradient,S,RN_dim)+RN_min(sparse_bilinear(S,RN_Hesse,S),0);
}

int is_at_boundary(double* X){
	const double eps = 1e-6;
	int i,index;
	for(i=0;i<RN_bound_num;i++){
		index = RN_bound[i];
		if (!isnan(RN_upper[i]) && fabs(X[index]-RN_upper[i])<eps) return 1;
		if (!isnan(RN_lower[i]) && fabs(X[index]-RN_lower[i])<eps) return 1;
	}
	return 0;
}

/*void set_aux_constraints(int* Indices, int size){
	RN_aux_constraints = Indices;
	RN_aux_size = size;
}

void include_aux_constraints(sparse_matrix* A,double* b,int* Indices,int size){
	int i,j,l,J,ind;
	if (Indices!=NULL) for (i=0;i<size;i++){
		ind = Indices[i];
		reset_row(A,ind);
		insert_sparse(A,1.,ind,ind);
		if (b!=NULL) b[ind] = 0.;
		for (l=0;l<A->size;l++) if (l!=ind){
			for (j=0;j<A->Len[l];j++){
				J = A->Indices[l][j];
				if (J==ind) remove_element_at(A,l,j);
			}
		}
	}
}*/

void RN_iterate(sparse_matrix* M,double* Gradient,double* D,double* X){
	const int max_iter = 10;
	
	int iter;
	double d,alpha,alpha_l,alpha_r,contraction;
	double* S = NULL;
	double* X_old = clone_vector(X,RN_dim);
	
	double* V1 = zero_vector(RN_dim);					// get subspace spanned by V1,V2
	double* V2 = zero_vector(RN_dim);
	RN_get_subspace(M,Gradient,D,X,V1,V2);
	double* B = clone_vector(Gradient,RN_dim);
	vector_pseudo_mult(B,D,RN_dim);
	
	if (RN_status==POSITIVE_DEFINITE && euklid_norm(V2,RN_dim)<RN_Delta) S = clone_vector(V2,RN_dim);
	else{
		vector_normalize(V1,RN_dim);				// orthonormalize basis V1,V2
		double s = scalar(V1,V2,RN_dim);
		vector_add(V2,V1,-s,RN_dim);
		vector_normalize(V2,RN_dim);
		
		S = zero_vector(RN_dim);					// get constrained minimum in subspace
		subspace_minimum(M,B,V1,V2,S);
	}
	vector_pseudo_mult(S,D,RN_dim);					// transform back 
	
	alpha = 1.0;										// update with alpha=1
	vector_add(X,S,alpha,RN_dim);					
	RN_reflect(X);
	
	
	if (!RN_test_sigma_l(X_old,X,S,Gradient,alpha)){	// line search 			(piecewise linear bisection algorithm)
		statistics.line_search_count++;
		iter = 0;
		alpha_l = 0;
		alpha_r = 1.;
		do{
			alpha = (alpha_l+alpha_r)/2.;
			copy_vector_to(X_old,X,RN_dim);
			vector_add(X,S,alpha,RN_dim);
			RN_reflect(X);
			if (!RN_test_sigma_l(X_old,X,S,Gradient,alpha)){
				alpha_r = alpha;
			}
			else{
				if (!RN_test_sigma_u(X_old,X,S,Gradient,alpha) && !(alpha>RN_rho)) alpha_l = alpha; else break;
			}
			iter++;
		}while(iter<max_iter);
		if (RN_info==RN_YES && iter==max_iter) printf("Reflective Newton warning: line search did not converge");
		statistics.line_search_iter += iter;
	}
	
	if (is_at_boundary(X)){								// test if boundary is touched, if so: shift a bit
		alpha -= RN_Xi*euklid_norm(B,RN_dim);
		copy_vector_to(X_old,X,RN_dim);
		vector_add(X,S,alpha,RN_dim);
		RN_reflect(X);
	}
	
	d = ((*RN_Functional)(X,RN_dim)-(*RN_Functional)(X_old,RN_dim));
	contraction = d/predicted_readuction(S,Gradient);
	if (contraction>RN_trust_upper && d<0){
		RN_Delta = RN_min(RN_trust_radius_max,2.*RN_Delta);
		if (RN_info==RN_YES) printf("reflective Newton: increase trust radius: %e\n",RN_Delta);
	}
	if (contraction<RN_trust_lower || d>=0){
		RN_Delta = RN_max(RN_trust_radius_min,RN_Delta/2.);
		if (RN_info==RN_YES) printf("reflective Newton: decrease trust radius: %e\n",RN_Delta);
	}
	if (RN_info==RN_YES) printf("contraction: %f\n",contraction);
	
	free(V1);
	free(V2);
	free(B);
	free(S);
	free(X_old);
}

void set_RN_params(int max_iter,double eps, double sigma_l, double sigma_u){
	RN_max_iter = max_iter;
	RN_eps = eps;
	RN_sigma_l = sigma_l;
	RN_sigma_u = sigma_u;
}

void set_RN_constraints(int* Index_set, double* Lower, double* Upper, int n){
	int i;
	RN_bound = Index_set;
	RN_bound_num = n;
	RN_lower = Lower;
	RN_upper = Upper;
	if (RN_info==YES){
		for (i=0;i<n;i++) if (Lower[i]==Upper[i]) printf("Reflective Newton: upper==lower at index %d -> use Dirichlet\n",i);
	}
}

void set_RN_functional(double (*F)(double* X,int n)){
	RN_Functional = F;
}

void set_RN_Hesse(sparse_matrix* A){				// here: q(x) = x.H.x/2+x.b
	RN_Hesse = A;
	if (A!=NULL) RN_dim = A->size;	
}

void set_RN_linear(double* B){						// only used when default functional,gradient
	RN_b = B;
}

int reflective_newton(double* X){
	double* D = NULL;
	double* G = NULL;
	int iter = 0;
	double q,dq;
	if (RN_Functional==NULL) RN_Functional = RN_default_func;
	if (RN_gradient==NULL) RN_gradient = RN_default_gradient;
	double q0 = (*RN_Functional)(X,RN_dim);
	do{
		if (RN_compute_Hesse!=NULL){
			if (RN_Hesse!=NULL){free_sparse(RN_Hesse);}
			RN_Hesse = RN_compute_Hesse(X);
		}
		D = zero_vector(RN_dim);
		G = (*RN_gradient)(X);		
		sparse_matrix* M = RN_get_M_D(G,X,D);
		RN_iterate(M,G,D,X);
		q = (*RN_Functional)(X,RN_dim);
		dq = fabs((q-q0)/q0);
		if (RN_info==RN_YES) printf("overall reduction = %e\n\n",dq);
		iter++;
		q0 = q;
		free_sparse(M);
		free(D);
		free(G);
	}while(iter<RN_max_iter && dq>RN_eps);
	if (iter==RN_max_iter) printf("Reflective Newton: no convergence after %d iterations\n",iter);
	statistics.res += dq;
	statistics.total_iter += iter;
	statistics.calls++;
	if (RN_compute_Hesse!=NULL && RN_Hesse!=NULL){free_sparse(RN_Hesse);}
	return iter;
}

int simulated_annealing_min(double* X,int n){
	int i,j,sign;
	double prev,curr,novel,p,a;
	double dx = (double)2/1000;
	double T = .0001;
	double z = (double)10/n;
	
	srand(time(NULL));
	if (RN_bound_num!=RN_dim){
		printf("not enough constraints -> abort\n");
		exit(0);
	}
	prev = (*RN_Functional)(X,RN_dim);
	printf("start value: %f\n",prev);
	do{
		do{
			j = rand() % RN_dim;
			sign = 1-2*(rand() % 2);
			novel = X[j]+sign*dx;
		}while(novel<RN_lower[j] || novel>RN_upper[j]);
		X[j] = novel;
		
		curr = (*RN_Functional)(X,RN_dim);
		a = exp((curr-prev)/T);
		if (a<1){
			p = (double)rand()/RAND_MAX;
			if (p<a) X[j] -= sign*dx; else prev = curr;
		}
		else X[j] -= sign*dx;
		i++;
		T *= (1.+z);
	}while(i<n);
	print_vector(X,RN_dim);
	printf("final value: %f\n",prev);
	return i;
}
