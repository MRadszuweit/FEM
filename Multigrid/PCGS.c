
#include "PCGS.h"

static int precon_maxiter;
static int (*precon)(int n,double* R,double* Z);
static void (*mat_multi)(int n,double* x,double* y);
static double pcg_residuum;

// for AMG preconditioner only

static sparse_matrix* PCG_A = NULL;
static sparse_matrix** PCG_L = NULL;
static sparse_matrix** PCG_U = NULL;
static sparse_matrix** PCG_Restriction = NULL;
static sparse_matrix** PCG_Prolongation = NULL;
static sparse_matrix** PCG_Coarse_A = NULL;
static double** PCG_C_L = NULL;
static double** PCG_C_U = NULL;
static int** PCG_Coarsening = NULL;
static int* PCG_Coarse_sizes = NULL;
static int* PCG_Fine_sizes = NULL;
static int* PCG_AMG_iter = NULL;
static int PCG_AMG_depth;

static double (*PCG_AMG_solver)(
		sparse_matrix** Coarse_A,
		sparse_matrix** Prolongation,
		sparse_matrix** Restriction,
		sparse_matrix** L,
		sparse_matrix** U,
		double** Coarsest_L,
		double** Coarsest_U,
		double* Sol,
		double* Fine_F,
		int** Coarsening,
		int* Fine_sizes,
		int* Coarse_sizes,
		int* Iterations,
		int max_depth,
		int depth
) = NULL;

void PCG_set_AMG_solver(double (*Prec_solver)(
		sparse_matrix** Coarse_A,
		sparse_matrix** Prolongation,
		sparse_matrix** Restriction,
		sparse_matrix** L,
		sparse_matrix** U,
		double** Coarsest_L,
		double** Coarsest_U,
		double* Sol,
		double* Fine_F,
		int** Coarsening,
		int* Fine_sizes,
		int* Coarse_sizes,
		int* Iterations,
		int max_depth,
		int depth)){
	PCG_AMG_solver = Prec_solver;
}

void PCG_set_AMG_system(
		sparse_matrix* A,
		sparse_matrix** L,
		sparse_matrix** U,
		sparse_matrix** Restriction,
		sparse_matrix** Prolongation,
		sparse_matrix** Coarse_A, 
		double** C_L,
		double** C_U,
		int** Coarsening,
		int* Coarse_sizes,
		int* Fine_sizes,
		int depth){
	int i;
	PCG_A = A;
	PCG_L = L;
	PCG_U = U;
	PCG_Restriction = Restriction;
	PCG_Prolongation = Prolongation;
	PCG_Coarse_A = Coarse_A;
	PCG_C_L = C_L;
	PCG_C_U = C_U;
	PCG_Coarsening = Coarsening;
	PCG_Coarse_sizes = Coarse_sizes;
	PCG_Fine_sizes = Fine_sizes;
	PCG_AMG_depth = depth;
	PCG_AMG_iter = (int*)malloc(depth*sizeof(int));
	for (i=0;i<depth;i++) PCG_AMG_iter[i] = 1;
	if (depth-1>=0) PCG_AMG_iter[depth-1] = 1;
	if (depth-2>=0) PCG_AMG_iter[depth-2] = 2;
}

int PCG_AMG_precon(int n,double* R,double* Z){
	//AMG_fix_isolated(PCG_A,R,Z,Coarsening[depth-1]);
	(*PCG_AMG_solver)(PCG_Coarse_A,PCG_Prolongation,PCG_Restriction,PCG_L,PCG_U,
	PCG_C_L,PCG_C_U,R,Z,PCG_Coarsening,PCG_Fine_sizes,PCG_Coarse_sizes,PCG_AMG_iter,PCG_AMG_depth,PCG_AMG_depth-1);
	return PCG_AMG_iter[PCG_AMG_depth-1];
}

void PCG_AMG_mat_mult(int n,double* x,double* y){
	copy_vector_to(x,y,n);
	sparse_multiplication(PCG_A,y);
}

void PCG_reset(){
	PCG_A = NULL;
	PCG_L = NULL;
	PCG_U = NULL;
	PCG_Restriction = NULL;
	PCG_Prolongation = NULL;
	PCG_Coarse_A = NULL;
	PCG_C_L = NULL;
	PCG_C_U = NULL;
	PCG_Coarsening = NULL;
	PCG_Coarse_sizes = NULL;
	PCG_Fine_sizes = NULL;
	free(PCG_AMG_iter);
	PCG_AMG_solver = NULL;
	precon = NULL;
	mat_multi = NULL;
}

// end AMG preconditioner

// normal PCG

void PCG_set_precon(int (*Precon)(int n,double* R,double* Z),int max_precon){
	precon = Precon;
	precon_maxiter = max_precon;
}

void* PCG_get_precon(){
	return precon;
}

void PCG_set_mat_mult(void (*Mat_mult)(int n,double* x,double* y)){
	mat_multi = Mat_mult;
}

void* PCG_get_mat_mult(){
	return mat_multi;
}

int PCG_get_precon_iter(){
	return precon_maxiter;
}

double PCG_residuum(int n,double* X,double* B){
	int i;
	double r;
	double* Y = zero_vector(n);
	double* R = zero_vector(n);
	(*mat_multi)(n,X,Y);
	for (i=0;i<n;i++) Y[i] -= B[i];
	(*precon)(n,B,R);
	r = euklid_norm(R,n);
	free(Y);
	free(R);
	return r;
}

int pcg(double* X,double* B,int n,double tol,int max_iter){
	int i,k;
	int flag = 0;
	double s,s_old,a,b,r,r0,h;
	
	h = euklid_norm(B,n);
	if (h==0){h = 1;}
	scalar_mult((double)1/h,B,n);
	scalar_mult((double)1/h,X,n);
	
	double* R;
	double* P;
	double* W = zero_vector(n);
	double* Q = zero_vector(n);
	double* Z = zero_vector(n);	
	R = clone_vector(B,n);
	(*mat_multi)(n,X,Q);
	vector_add(R,Q,-1.,n);
	if (PCG_A!=NULL) r0 = ILU_residuum(PCG_A,PCG_L[PCG_AMG_depth-1],PCG_U[PCG_AMG_depth-1],X,B);
	else{
		(*precon)(n,R,W);
		r0 = euklid_norm(W,n);
	}
	if (r0==0){
		free(Q);
		free(Z);
		return 0;
	}
	r = r0;
	i = 0;
	do{
		k = (*precon)(n,R,Z);
		if (k>=precon_maxiter && flag>=0){
			printf("Warnung: Maximale Anzahl von preconditioner Iterationen erreicht\n");
			flag = 1;
			/*printf("a = %lf\n",a);
			printf("s = %lf\n",s);
			dummy();*/
			return -1;
		}
		s_old = s;
		s = scalar(R,Z,n);
		if (i>0){
			b = s/s_old;
			scalar_mult(b,P,n);
			vector_add(P,Z,1,n);
		}
		else{P = clone_vector(Z,n);}
		(*mat_multi)(n,P,Q);
		a = s/scalar(P,Q,n);
		vector_add(X,P,a,n);
		vector_add(R,Q,-a,n);
		r = euklid_norm(R,n);
		i++;
		//printf("pcg res %f\n",r/r0);
	}while(r/r0>tol && i<max_iter);
	if (i==max_iter){
		//printf("Warnung: maximale Anzahl von %d Iterationen erreicht\n",i);
	}
	if (PCG_A!=NULL) r = ILU_residuum(PCG_A,PCG_L[PCG_AMG_depth-1],PCG_U[PCG_AMG_depth-1],X,B);
	else{
		(*precon)(n,R,W);
		r = euklid_norm(W,n);
	}
	//printf("Residuum pcg: %f\n",r/r0);
	if (isnan(r/r0)){
		printf("Fehler bei PCG: nan -> Abbruch\n");
		exit(0);
	}
	free(W);
	free(Q);
	free(Z);
	free(R);
	free(P);
	scalar_mult(h,B,n);
	scalar_mult(h,X,n);
	pcg_residuum = r/r0;
	return i;
}

double get_pcg_residuum(){
	return pcg_residuum;
}
