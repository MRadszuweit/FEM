#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "GMRES_Newton.c.opari.inc"
#line 1 "GMRES_Newton.c"

#include "GMRES_Newton.h"

// global variables:

double h = 1E-6;

static sparse_matrix* Newton_Matrix = NULL;
static sparse_matrix* Newton_L = NULL;
static sparse_matrix* Newton_U = NULL;
static sparse_matrix* Newton_Pattern = NULL;

static double** Newton_Roots = NULL;

static double* Newton_B = NULL;

static int Newton_root_number = 0;

static double* Newton_search_min = NULL;
static double* Newton_search_max = NULL;
static double* Newton_range_min = NULL;
static double* Newton_range_max = NULL;
static int Newton_guesses = 10000;

#ifndef __cplusplus

static struct ITLIN_OPT* Newton_Solver_options;
static struct ITLIN_INFO* Newton_Solver_info;

#endif

double (*Newton_F)(double* X,int i,int size);

// functions:

#ifndef __cplusplus

void Newton_GMRES_options(int bases_imax,int maxiter,double tol){
	Newton_Solver_options = malloc(sizeof(struct ITLIN_OPT));
	Newton_Solver_options->tol = tol;
	Newton_Solver_options->i_max = bases_imax;  
	Newton_Solver_options->maxiter = maxiter;
	Newton_Solver_options->termcheck = CheckOnRestart;
	
	Newton_Solver_options->errorlevel = Verbose;
	Newton_Solver_options->monitorlevel = None;
	Newton_Solver_options->datalevel = None;
	
	Newton_Solver_options->errorfile = stdout;
	Newton_Solver_options->monitorfile = stdout;
	Newton_Solver_options->datafile = NULL;
	Newton_Solver_options->iterfile = NULL;
	Newton_Solver_options->resfile = NULL;
	Newton_Solver_options->miscfile = NULL;
	
	Newton_Solver_info = malloc(sizeof(struct ITLIN_INFO));
}

#endif

void Newton_first_guess_options(int guess_num,int size,double* Min,double* Max){
	int i;
	Newton_guesses = guess_num;
	if (Newton_search_min==NULL) Newton_search_min = zero_vector(size);
	if (Newton_search_max==NULL) Newton_search_max = zero_vector(size);
	for (i=0;i<size;i++){
		Newton_search_min[i] = Min[i];
		Newton_search_max[i] = Max[i];
	}
}

void Newton_set_range(int size,double* Min,double* Max){
	int i;
	if (Newton_range_min==NULL) Newton_range_min = zero_vector(size);
	if (Newton_range_max==NULL) Newton_range_max = zero_vector(size);
	for (i=0;i<size;i++){
		Newton_range_min[i] = Min[i];
		Newton_range_max[i] = Max[i];
	}
}

void Newton_set_function(double (*F)(double* X,int i,int size)){
	Newton_F = F;
}

void Newton_set_inhomo(double* B,int n){
	int i;
	if (Newton_B==NULL) Newton_B = zero_vector(n);
	if (B!=NULL) for (i=0;i<n;i++) Newton_B[i] = B[i];
}

void Newton_matvec(int n,double* y,double* z){
	int i,j;
	double sum;
	sparse_matrix* A = Newton_Matrix;
	for (i=0;i<n;i++){
		sum = 0;
		for (j=0;j<A->Len[i];j++){sum += A->Values[i][j]*y[A->Indices[i][j]];}
		z[i] = sum;
	}
}

int Newton_preconlr(int n,double *z,double* w){
	copy_vector_content(z,w,0,0,n);
	L_triang_invert(Newton_L,w);
	U_triang_invert(Newton_U,w);
	return 1;
}

sparse_matrix* Newton_get_sparse_Jacobian(double (*F)(double* X,int i,int size),sparse_matrix* Pattern,
 double* X,double h){
	int i,j,k;
	double f0;
	int n = Pattern->size;
	sparse_matrix* Jacobian = sparse_zero(n);
	for (i=0;i<n;i++){
		f0 = (*F)(X,i,n);
		for (j=0;j<Pattern->Len[i];j++) if (Pattern->Values[i][j]!=0){
			k = Pattern->Indices[i][j];
			X[k] += h;
			insert_sparse(Jacobian,((*F)(X,i,n)-f0)/h,i,k);
			X[k] -= h;
		}
	}
	return Jacobian;
}

double Newton_residuum(double (*F)(double* X,int i,int size),double* X,double* B,int size){
	int i;
	double* Res = zero_vector(size);
	for (i=0;i<size;i++) Res[i] = B[i]-(*F)(X,i,size);
	free(Res);
	return euklid_norm(Res,size);
}

void Newton_set_Residuum(double (*F)(double* X,int i,int size),double* X,double* Y,
 double* B,int size){
	int i;
	for (i=0;i<size;i++) Y[i] = B[i]-(*F)(X,i,size);
}

double mod_F(double* X,int i,int size){
	int j,k,r;
	double sum;
	double pro = 1.;
	for (r=0;r<Newton_root_number;r++){
		sum = 0;
		for (j=0;j<Newton_Pattern->Len[i];j++){
			k = Newton_Pattern->Indices[i][j];
			sum += (X[k]-Newton_Roots[r][k])*(X[k]-Newton_Roots[r][k]);
		}
		pro *= sum;
	}
	return (*Newton_F)(X,i,size)/sqrt(pro);
}

double** Newton_get_all_roots(){
	int i;
	int n = Newton_Pattern->size;
	double** Res = (double**)malloc(Newton_root_number*sizeof(double*));
	for (i=0;i<Newton_root_number;i++){
		Res[i] = clone_vector(Newton_Roots[i],n);
		free(Newton_Roots[i]);
	}
	free(Newton_Roots);
	Newton_Roots = NULL;
	return Res;
}

sparse_matrix* Newton_get_full_pattern(int size){
	int i,j;
	sparse_matrix* Res = sparse_zero(size);
	for (i=0;i<size;i++){
		for (j=0;j<size;j++) insert_sparse(Res,1.,i,j);
	}
	return Res;
}

int Newton_in_range(double* X){
	int i;
	for (i=0;i<Newton_Pattern->size;i++){
		if (X[i]<Newton_range_min[i] || X[i]>Newton_range_max[i]){
			//printf("Newton Warning: out of range X%d=%f\n",i,X[i]);
			return 0;
		}
	}
	return 1;
}

void Newton_first_guess(double* Res,int n){
	int i,j;
	double r,r_min;
	double* Guess = zero_vector(n);
	double* Value = zero_vector(n);
	//srand(time(NULL));
	r_min = DBL_MAX;
	for (i=0;i<Newton_guesses;i++){
		for (j=0;j<n;j++) Guess[j] = (double)Newton_search_min[j]+rand()*(Newton_search_max[j]-Newton_search_min[j])/RAND_MAX;
		for (j=0;j<n;j++) Value[j] = mod_F(Guess,j,n)-Newton_B[j];
		r = euklid_norm(Value,n);
		if (r<r_min){
			r_min = r;
			copy_vector_content(Guess,Res,0,0,n);
		}
	}
	free(Guess);
	free(Value);
}

#ifndef __cplusplus

int Newton_solver(sparse_matrix* Pattern,double* X,double maxiter,double eps,int mode,double r0){
	double* H;
	double r,old_r;
	int n = Pattern->size;
	double* R = zero_vector(n);
	Newton_Pattern = Pattern;
	if (r0<0){
		Newton_set_Residuum(mod_F,X,R,Newton_B,n);
		r0 = euklid_norm(R,n);
	}
	r = r0;
	int i = 0;
	if (r0!=0) do{
		Newton_Matrix = Newton_get_sparse_Jacobian(mod_F,Newton_Pattern,X,h);
		Newton_L = sparse_zero(n);
		Newton_U = sparse_zero(n);
		if (mode==LARGE_SYSTEM){
			incomplete_LU_factorization(Newton_Matrix,Newton_L,Newton_U);
			H = zero_vector(n);
			gmres(n,H,&Newton_matvec,&Newton_preconlr,NULL,R,Newton_Solver_options,Newton_Solver_info);
			if (Newton_Solver_info->iter>=Newton_Solver_options->maxiter){
				i = maxiter;
				break;
			}
		}
		else{
			sparse_LU_factorization(Newton_Matrix,Newton_L,Newton_U);
			H = clone_vector(R,n);
		}
		L_triang_invert(Newton_L,H);
		U_triang_invert(Newton_U,H);
		vector_add(X,H,1.,n);
		Newton_set_Residuum(mod_F,X,R,Newton_B,n);
		old_r = r;
		r = euklid_norm(R,n);
		if (isnan(r)){
			printf("Newton Warning: singular Jacobian\n");
			printf("Newton residuum reduction: %e\n",r/r0);
			return -1;
		}
		//printf("newton residuum: %f at (%f,%f) val: %f\n",r/r0,X[0],X[1],mod_F(X,0,2));
		if (!Newton_in_range(X)){
			//printf("Newton X=(%f,%f,%f,%f) out of range -> new guess\n",X[0],X[1],X[2],X[3]);
			Newton_first_guess(X,n);
			Newton_set_Residuum(mod_F,X,R,Newton_B,n);
			r = euklid_norm(R,n);
			//i--;
		}
		free_sparse(Newton_Matrix);
		free_sparse(Newton_L);
		free_sparse(Newton_U);
		free(H);
		i++;
	}while(i<maxiter && r/r0 > eps);
	free(R);
	if (i==maxiter){
		printf("GMRES Newton: no convergence with given tolerance\n -> abort\n");
		return -1;
	}
	if (i!=0) printf("GMRES Newton: residuum reduction: %f ; %d iterations\n",r/r0,i);
	else printf("GMRES Newton: initial residuum equals zero -> continue ...\n");
	return i;
} 

int Newton_solver_mult_Root(sparse_matrix* Pattern,double maxiter,double eps,int mode){
	int iter;
	int n = Pattern->size;
	double* Guess = zero_vector(n);
	Newton_root_number = 0;
	double* R = zero_vector(n);
	Newton_set_Residuum(mod_F,Guess,R,Newton_B,n);
	do{
		Newton_first_guess(Guess,n);
		iter = Newton_solver(Pattern,Guess,maxiter,eps,mode,euklid_norm(R,n));
		if (iter>=0 && iter<maxiter){
			Newton_root_number++;
			Newton_Roots = (double**)realloc(Newton_Roots,Newton_root_number*sizeof(double*));
			Newton_Roots[Newton_root_number-1] = clone_vector(Guess,n);
		}
	}while(iter>=0 && iter<maxiter);
	free(Guess);
	free(R);
	return Newton_root_number;
}

#endif

double divide_by_roots(double* Roots,int n,double x){
	int i;
	double res = 1.;
	for (i=0;i<n;i++) res *= (x-Roots[i]);
	return 1./res;
}

double root_poly(double* Roots,int root_num,double x){
	int i;
	double res = 1.;
	for (i=0;i<root_num;i++) res *= (x-Roots[i]);
	return res;
}

int in_roots(double* Roots,int size,double x){
	const double eps = 1e-8;
	int i;
	for (i=0;i<size;i++) if (fabs(Roots[i]-x)<eps) return 1;
	return 0;
}

double der_F(double (*F)(double x),double x,int depth){
	const double dx = 1e-6;
	if (depth==0) return (*F)(x);
	else{
		return (der_F(F,x+dx,depth-1)-der_F(F,x-dx,depth-1))/(2.*dx);
	}
}

double der_poly(double* Roots,int root_num,double x,int depth){
	const double dx = 1e-6;
	if (depth==0) return root_poly(Roots,root_num,x);
	else{
		return (der_poly(Roots,root_num,x+dx,depth-1)-der_poly(Roots,root_num,x-dx,depth-1))/(2.*dx);
	}
}

double F_hospital(double (*F)(double x),double* Roots,int root_num,double x){
	//if (in_roots(Roots,root_num,x)){		
		const double eps = 1e-5;
		int depth = 0;
		double dp,df;
		do{
			dp = der_poly(Roots,root_num,x,depth);
			df = der_F(F,x,depth);
			depth++;
		}while(fabs(dp)<eps && fabs(df)<eps);
		return df/dp;
	
	//else return divide_by_roots(*Roots,root_num,x)*(*F)(x);
	
	
}

int Newton_1D(double (*F)(double x),double** Roots,double min,double max,int max_iter,double eps){
	int i;
	double f,x,dummy,x0,x1,f0,f1,F0,local_eps;
	int root_num = 0;
	
	do{
		local_eps = eps;
		x0 = min;
		x1 = max;
		f0 = F_hospital(F,*Roots,root_num,x0);
		f1 = F_hospital(F,*Roots,root_num,x1);
		if (f0!=0) F0 = f0;	else F0 = 1.;
		i = 0;
		do{
			/*while(fabs(f0-f1)<eps){
				double r = (double)rand()/RAND_MAX;
				x1 = min+(max-min)*r;							
				f1 = F_hospital(F,*Roots,root_num,x1);
			}*/
			
			x = x1-f1*(x1-x0)/(f1-f0);
			if (x<min || x>max){
				double r = (double)rand()/RAND_MAX;
				x = min+(max-min)*r;
			}		
			f = F_hospital(F,*Roots,root_num,x);
			x0 = x1;
			x1 = x;
			f0 = f1;
			f1 = f;
			//if (i>max_iter-10 && root_num==0) printf("%d: x=%f res=%f\n",i,x,fabs(f/F0));
			i++;
			if (i==max_iter/2) local_eps *= 1e2;									// lower precision if too many iterations are needed
		}while(fabs(f/F0)>local_eps && i<max_iter);
		if (i<max_iter){
			root_num++;
			*Roots = (double*)realloc(*Roots,root_num*sizeof(double));
			(*Roots)[root_num-1] = x;
			//printf("\n root %d: %f\n iterations: %d reduction: %f\n\n",root_num,x,i,f/F0);			
		}
	}while(i<max_iter);
	if (root_num==0){
		printf("Newton 1D: no roots found in given interval\n");
		int i;
		
	}
	return root_num;
}

