
#include "BiCGStab.h"

//////////////////////////////// global variables //////////////////////////////////////////

static int (*Precon)(int n,double* Y,double* X);
static void (*Mat_mult)(int n,double* Y,double* X);
static amg_system_info* BI_AMG_data = NULL;
static amg_statistics BI_AMG_stat;
static amg_statistics BI_stat;

static int BI_max_iter = 1000;
static double BI_eps = 1e-8;
static int BI_info = YES;

//////////////////////////////// functions /////////////////////////////////////////////////


void BI_set_precon(int (*Preconditioner)(int n,double* Y,double* X)){ 		//X = M^{-1}.Y
	Precon = Preconditioner;
}

void BI_set_mat_mult(void (*Matrix_multiplication)(int n,double* Y,double* X)){ 			//  X = A.Y
	Mat_mult = Matrix_multiplication;
}

amg_system_info* BI_get_AMG_setup_data(){
	return BI_AMG_data;
}

void BI_set_AMG_setup_data(amg_system_info* AMG_data){
	BI_AMG_data = AMG_data;
}

void BI_set_eps(double eps){
	BI_eps = eps;
}

void BI_set_max_iter(int max_iter){
	BI_max_iter = max_iter;
}

int BI_get_max_iter(){
	return BI_max_iter;
}

void BI_set_stat(char* Out_AMG,char* Out_BiCGStab){
	BI_AMG_stat = AMG_init_stat(Out_AMG);		
	BI_stat = AMG_init_stat(Out_BiCGStab);		
}

amg_statistics* BI_get_AMG_stat(){
	return &BI_AMG_stat;	
}

amg_statistics* BI_get_stat(){
	return &BI_stat;
}

void BI_print_info(int mode){
	BI_info = mode;
}

int BI_info_set(){
	return BI_info;
}

double BI_residuum(double* X,double* b,int n){
	double* Res = zero_vector(n);
	(*Mat_mult)(n,X,Res);
	vector_add(Res,b,-1.,n);
	scalar_mult(-1,Res,n);
	return euklid_norm(Res,n);
}

int BI_PCG_precon(int n,double* Y,double* X){
	const double eps = 1e-2;
	const int max_iter = 1000;
	int i = pcg(X,Y,n,eps,max_iter);
	if (i==max_iter) printf("warning PCG preconditioner could not solve to required residuum reduction\n");
	return i;
}

int BI_AMG_precon(int n,double* Y,double* X){
	return AMG_solve(BI_AMG_data,Y,X,&BI_AMG_stat);	
}

void BI_set_PCG_preconditioner(sparse_matrix* A){
	int n = A->size;
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(A,L,U);
	PCG_default(A,L,U);
	PCG_set_default_precon();
	BI_set_precon(&BI_PCG_precon);
}

void BI_set_AMG_preconditioner(sparse_matrix* A,int num_fields,int depth){
	const double eps = 1e-4;				// residuum reduction for each preconditer step
	const double tol = 1e-1;				// elements smaller than this treshold are removed in Prolongation/Restriction
	const int smiter = 2;					// smooting iteration number
	const int max_iter = 20;				// maximum namber of V cycles in AMG
	const int coarsening = AGGRESSIVE;
	AMG_print_info(NO);
	//AMG_set_SOR_relaxation_coeff(1.6);	// only important if SOR smoother used
	AMG_set_matrix_tolerance(tol);
	AMG_set_max_iter(max_iter);
	AMG_set_eps(eps);
	printf("set AMG preconditioner: setup ...\n");
	BI_AMG_data = AMG_setup(A,num_fields,depth,smiter,smiter+1,coarsening,STAND_ALONE,DIRECT,NULL,NULL);	
	BI_set_precon(&BI_AMG_precon);
	printf("finished\n\n");
}

int BiCGStab_solve(double* X,double* b,int n){
	const double default_eps = 1e-10;
	int iter,i;
	double h,g,s,r,r0,t,alpha,beta,omega;	
	double* S = zero_vector(n);
	double* PS = zero_vector(n);		// preconditioned S
	double* PP = zero_vector(n);		// preconditioned P
	double* APP = zero_vector(n);
	double* APS = zero_vector(n);

    double* R = zero_vector(n);		// initial residuum
	(*Mat_mult)(n,X,R);
	vector_add(R,b,-1.,n);
	scalar_mult(-1,R,n);	
	
	double* R0 = clone_vector(R,n);	// set R0 with R.R0!=0 
	
	double* P = clone_vector(R,n);	// intial P=R
	
	iter = 0;
	r0 = euklid_norm(R,n);
	if (r0==0 && BI_info==BI_YES) printf("initial residuum zero -> continue ...\n");
	else{
		do{
			(*Precon)(n,P,PP);
			/*i = (*Precon)(n,P,PP);
			if (i==200){
				print_sparse(BI_AMG_data->Coarse_A[BI_AMG_data->depth-1]);
				exit(0);
			}*/
			(*Mat_mult)(n,PP,APP);
			h = scalar(R,R0,n);
			t = scalar(APP,R0,n);
			if (t==0){
				if (fabs(r0)>default_eps) printf("Warning: BiCGStab stopped because alpha->infinity\n");
				break;
			}
			
			alpha = h/t;									// alpha = R.R0/((A.PP).R0)
				
			copy_vector_to(R,S,n);							// S -> R-alpha*A.PP			
			vector_add(S,APP,-alpha,n);
			
			(*Precon)(n,S,PS);
			(*Mat_mult)(n,PS,APS);
			s = scalar(APS,APS,n);
			if (s==0){
				if (fabs(r0)>default_eps) printf("Warning: BiCGStab stopped because APS=0\n");
				break;
			}
			omega = scalar(APS,S,n)/s;		// omega = ((A.PS).S)/((A.PS).(A.PS))
			
			vector_add(X,PP,alpha,n);
			vector_add(X,PS,omega,n);						// X -> X+alpha*PP+omega*PS
			
			copy_vector_to(S,R,n);							// R -> S-omega*(A.PS)
			vector_add(R,APS,-omega,n);
			
			beta = scalar(R,R0,n)*alpha/(h*omega);			// beta = (R.R0)/(R_old.R0)*alpha/omega
			
			
			vector_add(P,APP,-omega,n);						// P -> R+beta*(P-omega*(A.PP))
			scalar_mult(beta,P,n);
			vector_add(P,R,1.,n);			
			
			r = euklid_norm(R,n);
			iter++;
			if (BI_info==BI_YES) printf("\tBiCGStab residuum: %e\n",r/r0);
		}while(r/r0>BI_eps && iter<=BI_max_iter);	
	}
	if (iter==BI_max_iter) printf("BiCGStab: no convergence in %d iterations\n\n",iter);
	AMG_add_stat(&BI_stat,iter,fabs(r/r0));
	
	free(S);								  // clean
	free(PS);
	free(P);
	free(PP);
	free(R);	
	free(R0);
	free(APP);
	free(APS);	
	
	return iter;
}
