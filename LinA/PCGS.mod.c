#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "PCGS.c.opari.inc"
#line 1 "PCGS.c"

#include "PCGS.h"

extern int omp_chunk_size;

static int (*precon)(int n,double* R,double* Z);
static void (*mat_multi)(int n,double* x,double* y);
static double pcg_residuum;
static sparse_matrix* default_A = NULL;
static sparse_matrix* default_L = NULL;
static sparse_matrix* default_U = NULL;

// for AMG preconditioner only

static amg_system_info* PCG_AMG_data = NULL;

void PCG_reset(){
	default_A = NULL;
	default_L = NULL;
	default_U = NULL;
	precon = NULL;
	mat_multi = NULL;
}

// end AMG preconditioner

// normal PCG

void PCG_default(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	default_A = A;
	default_L = L;
	default_U = U;
	PCG_set_default_precon();
}

void PCG_mult(int n,double* y,double* z){ 									 //computes z=default_A.y
	if (default_A!=NULL){
		int i,j;
		double sum;
		int n = default_A->size;
		#ifdef _OPENMP
#line 42 "PCGS.c"
		#pragma omp parallel private(j,sum) firstprivate(default_A)
		{
#line 44 "PCGS.c"
			#pragma omp for schedule(dynamic,omp_chunk_size)
			for (i=0;i<n;i++){
				sum = 0;
				for (j=0;j<default_A->Len[i];j++){sum += default_A->Values[i][j]*y[default_A->Indices[i][j]];}
				z[i] = sum;
			}
		}
		#else
			for (i=0;i<n;i++){
				sum = 0;
				for (j=0;j<default_A->Len[i];j++){sum += default_A->Values[i][j]*y[default_A->Indices[i][j]];}
				z[i] = sum;
			}
		#endif
	}
}

void PCG_preconl(int n,double* z, double* w){
	int i,j,m;
	double sum;
	int* Ind;
	double* Val;
	for (i=0;i<n;i++){
		sum = z[i];
		m = default_L->Len[i]-1;
		Ind = default_L->Indices[i];
		Val = default_L->Values[i];
		for (j=0;j<m;j++){
			sum -= Val[j]*w[Ind[j]];
		}
		sum /= Val[m];
		w[i] = sum;
	}
}

void PCG_preconr(int n,double* z, double* w){
	int i,j,m;
	double sum,d;
	int* Ind;
	double* Val;
	for (i=n-1;i>=0;i--){
		sum = z[i];
		m = default_U->Len[i]-1;
		Ind = default_U->Indices[i];
		Val = default_U->Values[i];
		d = (*Val);
		Val++;
		Ind++;
		for (j=1;j<=m;j++){			
			sum -= (*Val)*w[*Ind];
			Val++;
			Ind++;
		}		
		sum /= d;
		w[i] = sum;
	}
}

int PCG_preconlr(int n,double *z,double* w){
	double* y = clone_vector(z,n);
	PCG_preconl(n,z,y);
	PCG_preconr(n,y,w);
	free(y);
	return 1;
}

void PCG_set_precon(int (*Precon)(int n,double* R,double* Z)){
	precon = Precon;
}

int PCG_AMG_precon(int n,double* Y,double* X){
	return AMG_solve(PCG_AMG_data,Y,X,NULL);	
}

void PCG_set_AMG_preconditioner(sparse_matrix* A,int num_fields,int depth){
	const double eps = 1e-2;
	const int smiter = 2;
	const int max_iter = 100;
	AMG_print_info(NO);
	AMG_set_SOR_relaxation_coeff(1.6);
	AMG_set_eps(eps);
	AMG_set_max_iter(max_iter);
	printf("set AMG preconditioner: setup ...\n");
	PCG_AMG_data = AMG_setup(A,num_fields,depth,smiter,smiter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);	
	PCG_set_precon(&PCG_AMG_precon);
}

void* PCG_get_precon(){
	return (void*)precon;
}

void PCG_set_mat_mult(void (*Mat_mult)(int n,double* x,double* y)){
	mat_multi = Mat_mult;
}

void* PCG_get_mat_mult(){
	return (void*)mat_multi;
}

void PCG_set_default_precon(){
	PCG_set_mat_mult(&PCG_mult);
	PCG_set_precon(&PCG_preconlr);
	
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
	(*precon)(n,R,W);
	r0 = euklid_norm(W,n);
	if (r0==0){
		free(Q);
		free(Z);
		return 0;
	}
	
	r = r0;
	i = 0;
	do{
		k = (*precon)(n,R,Z);
		/*if (k>=precon_maxiter && flag>=0){
			printf("Warnung: Maximale Anzahl von preconditioner Iterationen erreicht\n");
			flag = 1;
			return -1;
		}*/
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
		//printf("pcg res %e\n",r/r0);
	}while(r/r0>tol && i<max_iter);
	if (i==max_iter){
		printf("Warnung: maximale Anzahl von %d Iterationen erreicht\n",i);
	}
	(*precon)(n,R,W);
	r = euklid_norm(W,n);
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
