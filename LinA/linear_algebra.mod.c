#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "linear_algebra.c.opari.inc"
#line 1 "linear_algebra.c"

#include "linear_algebra.h"

// global variables 

int lg_thread_num = 1;
int omp_chunk_size = 1000;

// Code ///////////////////////////////////////////////

#ifndef __cplusplus

int min(int a,int b){
	return (a<b?a:b);
}

int max(int a,int b){
	return (a>=b?a:b);
}

#endif

void set_thread_num(int n){
	#ifdef _OPENMP
	if (n==AUTOMATIC){
		lg_thread_num = omp_get_num_procs();
	} 
	else lg_thread_num = n;
	omp_set_num_threads(lg_thread_num);
	printf("using %d threads\n",lg_thread_num);
	#endif 
}

void set_omp_chunk_size(int chunk){
	omp_chunk_size = chunk;
}

void test_omp(int size){
	int i,m;
	double x;
	double* X = zero_vector(size);
	#ifdef _OPENMP	
	const int chunk = 100;
	
#line 45 "linear_algebra.c"
	#pragma omp parallel private(i,x) shared(X)
	{
#line 47 "linear_algebra.c"
		#pragma omp for schedule(static)
		for (i=0;i<size;i++){
			x = (double)i/size;
			X[i] = sin(x)+sin(2.*x)+sin(3.*x)+sin(4.*x)+sin(5.*x);
		}
	}
	#endif
}

double* clone_vector(double* x,int n){
	int i;
	double* res = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){res[i] = x[i];}
	return res;
}

void copy_vector_to(double* x,double* y,int n){
	int i;
	#ifdef _OPENMP
#line 66 "linear_algebra.c"
	#pragma omp parallel private(i)
	{
#line 68 "linear_algebra.c"
		#pragma omp for schedule(dynamic,omp_chunk_size)
		for (i=0;i<n;i++) y[i] = x[i];
	}
	#else
	for (i=0;i<n;i++) y[i] = x[i];
	#endif
	
}

void copy_vector_content(double* x,double* y,int x_pos,int y_pos,int size){
	int i;
	for (i=x_pos;i<x_pos+size;i++) y[y_pos-x_pos+i] = x[i];
}

int* clone_list(int* x,int n){
	int i;
	int* res = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++){res[i] = x[i];}
	return res;
}

double* sparse_mult(sparse_matrix* A,double* x){
	int i,j;
	int n = A->size;
	double sum;
	double* y = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		sum = 0; 
		for (j=0;j<(A->Len[i]);j++){
			sum += A->Values[i][j]*x[A->Indices[i][j]];
		}
		y[i] = sum;
	}
	return y;
}

void linear_map(double* x,double fac,sparse_matrix* A,double* y){ // berechnet x+fac*A*y
	int i,j;
	int n = A->size;
	double sum;
	
	#ifdef _OPENMP
#line 110 "linear_algebra.c"
	#pragma omp parallel private(i,j,sum)
	{
#line 112 "linear_algebra.c"
		#pragma omp for schedule(dynamic,omp_chunk_size)
		for (i=0;i<n;i++){
			sum = 0; 
			for (j=0;j<(A->Len[i]);j++){sum += A->Values[i][j]*y[A->Indices[i][j]];}
			x[i] += fac*sum;
		}
	}
	#else
	for (i=0;i<n;i++){
		sum = 0; 
		for (j=0;j<(A->Len[i]);j++){sum += A->Values[i][j]*y[A->Indices[i][j]];}
		x[i] += fac*sum;
	}
	#endif
}

void sparse_multiplication(sparse_matrix* A,double* x){
	int i,j;
	double sum;
	int n = A->size;
	double* y = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		sum = 0;
		for (j=0;j<(A->Len[i]);j++){sum += A->Values[i][j]*x[A->Indices[i][j]];}
		y[i] = sum;
	}
	copy_vector_to(y,x,n);
	free(y);
}

sparse_matrix* sparse_product(sparse_matrix* A,sparse_matrix* B){ 
	int i,j,k,ind_k;
	double a,b;
	int n = A->size;
	sparse_matrix* P = sparse_zero(n);
	
	#ifdef _OPENMP
#line 149 "linear_algebra.c"
	#pragma omp parallel private(i,j,k,ind_k,a,b)
	{
#line 151 "linear_algebra.c"
		#pragma omp for schedule(dynamic,omp_chunk_size)
		for (i=0;i<n;i++){
			for (k=0;k<A->Len[i];k++){
				ind_k = A->Indices[i][k];
				a = A->Values[i][k];			
				if (a!=0) for (j=0;j<B->Len[ind_k];j++){
					b = B->Values[ind_k][j];
					if (b!=0) insert_sparse(P,a*b,i,B->Indices[ind_k][j]);
				}
			}
		}
	}
	#else
	for (i=0;i<n;i++){
		for (k=0;k<A->Len[i];k++){
			ind_k = A->Indices[i][k];
			a = A->Values[i][k];			
			if (a!=0) for (j=0;j<B->Len[ind_k];j++){
				b = B->Values[ind_k][j];
				if (b!=0) insert_sparse(P,a*b,i,B->Indices[ind_k][j]);				
			}
		}
	}
	#endif
	
	delete_zeros(P);
	return P;
}

sparse_matrix* serial_sparse_product(sparse_matrix* A,sparse_matrix* B){ //P = A*B
	int i,j,k,ind_k;
	double a,b;
	int n = A->size;
	sparse_matrix* P = sparse_zero(n);
	
	for (i=0;i<n;i++){
		for (k=0;k<A->Len[i];k++){			
			ind_k = A->Indices[i][k];
			a = A->Values[i][k];			
			if (a!=0) for (j=0;j<B->Len[ind_k];j++){
				b = B->Values[ind_k][j];
				if (b!=0) insert_sparse(P,a*b,i,B->Indices[ind_k][j]);				
			}
		}
	}
	delete_zeros(P);
	return P;
}

void sparse_left_mult(sparse_matrix* A, sparse_matrix* B){  // B->A*B
	sparse_matrix* P = sparse_product(A,B);
	copy_sparse_content(B,P);
	free_sparse(P);
}

void sparse_right_mult(sparse_matrix* A, sparse_matrix* B){  // A->A*B
	sparse_matrix* P = sparse_product(A,B);
	copy_sparse_content(A,P);
	free_sparse(P);
}

double scalar(double* x1,double* x2,int n){
	int i;
	double res = 0;
	for (i=0;i<n;i++){
		res = res + x1[i]*x2[i];
	}
	return res;
}

double euklid_norm(double* x1,int n){
	double  s = scalar(x1,x1,n);
	return sqrt(s);
}

double vec_dist(double* x1,double* x2,int n){
	int i;
	double d;
	double sum = 0;
	for (i=0;i<n;i++){
		d = x1[i]-x2[i];
		sum = sum + d*d;
	}
	return sqrt(sum);
}

void scalar_mult(double a,double* x,int n){
	int i;
	for (i=0;i<n;i++){
		x[i] *= a;
	}
}

void vector_normalize(double* x,int n){
	double a = euklid_norm(x,n);
	scalar_mult((double)1/a,x,n);
}

void vector_add(double* x,double* y,double factor,int n){
	int i;
	for (i=0;i<n;i++){
		x[i] += factor*y[i];
	}
}

void scalar_sparse_mult(double a,sparse_matrix* A){
	int i,j;
	for (i=0;i<A->size;i++){
		for (j=0;j<(A->Len[i]);j++){A->Values[i][j] *= a;}
	}
}

void sparse_add(sparse_matrix* A,sparse_matrix* B,double factor){
	int i,j;	
	if (B!=NULL){
		for (i=0;i<A->size;i++){
			for (j=0;j<(B->Len[i]);j++){
				insert_sparse(A,factor*B->Values[i][j],i,B->Indices[i][j]);
			}
		}
	}
}

void vector_pseudo_mult(double* x,double* y,int n){
	int i;
	for (i=0;i<n;i++) x[i] *= y[i];
}

void vector_shift(double* x,int n,double a){
	int i;
	for (i=0;i<n;i++) x[i] += a;
}

void partial_vector_shift(double* x,int start,int end,double a){
	int i;
	for (i=start;i<end;i++) x[i] += a;
}

double sparse_bilinear(double* y,sparse_matrix* A,double* x){
	double* Ax = sparse_mult(A,x);
	double res = scalar(y,Ax,A->size);
	free(Ax);
	return res;
}

void normalize_row_sum(sparse_matrix* A){
	int i,j;
	double sum;
	for (i=0;i<A->size;i++){
		sum = 0;
		for (j=0;j<A->Len[i];j++) sum += A->Values[i][j];
		if (sum!=0) for (j=0;j<A->Len[i];j++) A->Values[i][j] /= sum;
	}
}

sparse_matrix* sparse_diagonal(double* d,int n){
	int i;
	sparse_matrix* res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	res->Values = (double**)malloc(n*sizeof(double*));
	res->Indices = (int**)malloc(n*sizeof(int*));
	res->Len = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++){
		res->Values[i] = (double*)malloc(sizeof(double));
		res->Indices[i] = (int*)malloc(sizeof(int));
		res->Values[i][0] = d[i];
		res->Indices[i][0] = i;
		res->Len[i] = 1;
	}
	res->size = n;
	return res;
}

sparse_matrix* sparse_zero(int length){
	int i;
	double* zero = (double*)malloc(length*sizeof(double));
	for (i=0;i<length;i++){zero[i] = 0;}
	sparse_matrix* Res = sparse_diagonal(zero,length);
	free(zero);
	return Res;
}

sparse_matrix3D* sparse_zero3D(int n){
	int i;
	sparse_matrix3D* res = (sparse_matrix3D*)malloc(sizeof(sparse_matrix3D));
	res->Values = (double**)malloc(n*sizeof(double*));
	res->Indices1 = (int**)malloc(n*sizeof(int*));
	res->Indices2 = (int**)malloc(n*sizeof(int*));
	res->Len = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++){
		res->Values[i] = NULL;
		res->Indices1[i] = NULL;
		res->Indices2[i] = NULL;
		res->Len[i] = 0;
	}
	res->size = n;
	return res;
}

sparse_matrix* sparse3D_vB(sparse_matrix3D* B,double* X,int new_size){   	// Xa.Babc
	int i,j;
	int n = B->size;
	sparse_matrix* Res = sparse_zero(new_size);
	for(i=0;i<n;i++){
		for(j=0;j<B->Len[i];j++) insert_sparse(Res,X[i]*B->Values[i][j],B->Indices1[i][j],B->Indices2[i][j]);
	}
	return Res;
}

sparse_matrix* sparse3D_Bv_symmetric(sparse_matrix3D* B,double* X){ // (Babc.Xb+Babc.Xc)/2
	int i,j;
	int n = B->size;
	sparse_matrix* Res = sparse_zero(n);
	for(i=0;i<n;i++){
		for(j=0;j<B->Len[i];j++){
			insert_sparse(Res,B->Values[i][j]*X[B->Indices2[i][j]]/2.,i,B->Indices1[i][j]);
			insert_sparse(Res,B->Values[i][j]*X[B->Indices1[i][j]]/2.,i,B->Indices2[i][j]);
		}
	}
	return Res;
	
}

sparse_matrix* sparse3D_Bv(sparse_matrix3D* B,double* X){				   	// Babc.Xc
	int i,j;
	int n = B->size;
	sparse_matrix* Res = sparse_zero(n);
	for(i=0;i<n;i++){
		for(j=0;j<B->Len[i];j++){
			insert_sparse(Res,B->Values[i][j]*X[B->Indices2[i][j]],i,B->Indices1[i][j]);
		}
	}
	return Res;
}

sparse_matrix* sparse3D_Bv_middle(sparse_matrix3D* B,double* X){   	// Babc.Xb
	int i,j;
	int n = B->size;
	sparse_matrix* Res = sparse_zero(n);
	for(i=0;i<n;i++){
		for(j=0;j<B->Len[i];j++) insert_sparse(Res,B->Values[i][j]*X[B->Indices1[i][j]],i,B->Indices2[i][j]);
	}
	return Res;
}

double* sparse3D_Bvv(sparse_matrix3D* B,double* X1,double* X2){				// Babc.(X1b X2c)
	int i,j;
	double sum;
	int n = B->size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++){
		sum = 0;
		for(j=0;j<B->Len[i];j++){
			if (B->Indices1[i][j]>=n/2 || B->Indices1[i][j]<0) printf("bad index1 at %d: %d\n",i,B->Indices1[i][j]);
			if (B->Indices2[i][j]>=n || B->Indices2[i][j]<0) printf("bad index2 at %d: %d\n",i,B->Indices2[i][j]);
			sum +=  B->Values[i][j]*X1[B->Indices1[i][j]]*X2[B->Indices2[i][j]];
		}
		Res[i] = sum;
	}
	return Res;
}

void sparse3D_scalar_mult(sparse_matrix3D* B,double a){
	int i,j;
	double sum;
	int n = B->size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++){
		for(j=0;j<B->Len[i];j++) B->Values[i][j] *= a;
	}
}

void sparse3D_add(sparse_matrix3D* A,sparse_matrix3D* B){						// A+B -> A, slow version, not ordered
	int i,j,J,k,m,ind1,ind2;
	int n = A->size;
	for (i=0;i<n;i++){
		for (k=0;k<B->Len[i];k++){
			ind1 = B->Indices1[i][k];
			ind2 = B->Indices2[i][k];			
			J = -1;
			for (j=0;j<A->Len[i];j++){
				if (ind1==A->Indices1[i][j] && ind2==A->Indices2[i][j]){
					J = j;
					break;
				}
			}
			if (J<0){
				A->Len[i]++;
				m = A->Len[i];
				A->Values[i] = (double*)realloc(A->Values[i],m*sizeof(double*));
				A->Indices1[i] = (int*)realloc(A->Indices1[i],m*sizeof(int*));
				A->Indices2[i] = (int*)realloc(A->Indices2[i],m*sizeof(int*));	
				A->Values[i][m-1] = B->Values[i][k];
				A->Indices1[i][m-1] = B->Indices1[i][k];
				A->Indices2[i][m-1] = B->Indices2[i][k];
			}
			else A->Values[i][J] += B->Values[i][k];
		}
		
	}
}

double* sparse3D_bilinear(sparse_matrix3D* B,double* X1,double* X2){
	int i,j;
	int n = B->size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++){
		for (j=0;j<B->Len[i];j++) Res[i] += B->Values[i][j]*X1[B->Indices1[i][j]]*X2[B->Indices2[i][j]];
	}
	return Res;
}

sparse_matrix* get_dependence_pattern3D(sparse_matrix3D* A){
	int i,j,s1,s2;
	int n = A->size;
	sparse_matrix* P = sparse_zero(n);
	for (i=0;i<n;i++){
		for (j=0;j<A->Len[i];j++){
			s1 = A->Indices1[i][j];
			s2 = A->Indices2[i][j];
			if (A->Values[i][j]!=0){
				insert_sparse(P,1.,i,s1);
				insert_sparse(P,1.,i,s2);
			}
		}
	}
	return P;
}

sparse_matrix* get_dependence_pattern(sparse_matrix* A){
	int i,j,s;
	int n = A->size;
	sparse_matrix* P = sparse_zero(n);
	for (i=0;i<n;i++){
		for (j=0;j<A->Len[i];j++){
			s = A->Indices[i][j];
			if (A->Values[i][j]!=0) insert_sparse(P,1.,i,s);
		}
	}
	return P;
}

sparse_matrix* sparse_identity(int length){
	int i;
	double* V = (double*)malloc(length*sizeof(double));
	for (i=0;i<length;i++){V[i] = 1;}
	sparse_matrix* Res = sparse_diagonal(V,length);
	free(V);
	return Res;
}

sparse_matrix* sparse_partial_identity(int length,int start,int end){
	int i;
	double* V = (double*)malloc(length*sizeof(double));
	for (i=0;i<length;i++){V[i] = 0;}
	for (i=start;i<end;i++){V[i] = 1;}
	sparse_matrix* Res = sparse_diagonal(V,length);
	free(V);
	return Res;
}

void delete_zeros(sparse_matrix* A){
	int i,j,len;
	for (i=0;i<A->size;i++){		
		if (A->Len[i]>1){
			j = 0;
			do{
				len = A->Len[i];
				if (A->Values[i][j]==0){remove_element_at(A,i,j);}
				if (A->Len[i]==len) j++;				
			}while(j<A->Len[i]);
		}
	}	
}

void Free_sparse(sparse_matrix* A){
	int i;
	for (i=0;i<A->size;i++){
		free(A->Values[i]);
		free(A->Indices[i]);
	}
	free(A->Len);
	free(A->Values);
	free(A->Indices);
}

void Free_sparse3D(sparse_matrix3D* A){
	int i;
	for (i=0;i<A->size;i++){
		if (A->Values[i]!=NULL) free(A->Values[i]);
		if (A->Indices1[i]!=NULL) free(A->Indices1[i]);
		if (A->Indices2[i]!=NULL) free(A->Indices2[i]);
	}
	free(A->Len);
	free(A->Values);
	free(A->Indices1);
	free(A->Indices2);
}


int write_sparse(sparse_matrix* A,char* path,char* name){
	int i;
	char Name[512];
	sprintf(Name,"%s/%s",path,name);
	FILE* file = fopen(Name,"w");
	int n = A->size;
	if (fwrite(&n,sizeof(int),1,file)!=1) return 0;
	if (fwrite(A->Len,sizeof(int),n,file)!=n) return 0;
	for (i=0;i<n;i++) if(fwrite(A->Indices[i],sizeof(int),A->Len[i],file)!=A->Len[i]) return 0;
	for (i=0;i<n;i++) if(fwrite(A->Values[i],sizeof(double),A->Len[i],file)!=A->Len[i]) return 0;
	fclose(file);
	return 1;
}

sparse_matrix* read_sparse(char* path,char* name){
	int i;
	char Name[512];
	sparse_matrix* Res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	sprintf(Name,"%s/%s",path,name);
	FILE* file = fopen(Name,"r");
	if (fread(&(Res->size),sizeof(int),1,file)!=1) return NULL;
	int n = Res->size;
	Res->Len = (int*)malloc(n*sizeof(int));
	if (fread(Res->Len,sizeof(int),n,file)!=n) return NULL;
	Res->Indices = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++){
		Res->Indices[i] = (int*)malloc((Res->Len[i])*sizeof(int));
		if (fread(Res->Indices[i],sizeof(int),Res->Len[i],file)!=Res->Len[i]) return NULL;
	}
	Res->Values = (double**)malloc(n*sizeof(double*));
	for (i=0;i<n;i++){
		Res->Values[i] = (double*)malloc((Res->Len[i])*sizeof(double));
		if (fread(Res->Values[i],sizeof(double),Res->Len[i],file)!=Res->Len[i]) return NULL;
	}
	fclose(file);
	return Res;
}

sparse_matrix* read_sparse_textfile(char* path,char* name){					// Achtung: ist buffersize in split() nicht 
	int i,j,k;																// groß genug werden die letzten Elemente nich mitgenommen
	size_t len;
	const int buff_size = 64000;
	int* Len;
	int* Sub_Len;
	char** Parts;
	char** Sub_Parts;
	char Line[buff_size];
	char Name[512];
	sprintf(Name,"%s/%s",path,name);
	FILE* file = fopen(Name,"r");
	int size = 0;
	while(readLine(file,Line,buff_size)){
		size++;
	}
	rewind(file);
	sparse_matrix* Res = sparse_zero(size);
	Len = (int*)malloc(sizeof(int));
	Sub_Len = (int*)malloc(sizeof(int));
	for (i=0;i<size;i++){
		readLine(file,Line,buff_size);
		Parts = split(Line," ",Len);
		for (j=0;j<(*Len);j++){
			Sub_Parts = split(Parts[j],"|",Sub_Len);
			if ((*Sub_Len)==2){
				len = strlen(Sub_Parts[0]);
				for (k=1;k<=len;k++) Sub_Parts[0][k-1] = Sub_Parts[0][k];
				len = strlen(Sub_Parts[1]);
				Sub_Parts[1][len-1] ='\0';
				insert_sparse(Res,atof(Sub_Parts[1]),i,atoi(Sub_Parts[0]));
			}
			if (Sub_Parts!=NULL){
				for (k=0;k<(*Sub_Len);k++) free(Sub_Parts[k]);
				free(Sub_Parts);
			}
		}
		if (Parts!=NULL){
			for (j=0;j<(*Len);j++) free(Parts[j]);
			free(Parts);
		}
	}
	fclose(file);
	free(Sub_Len);
	free(Len);
	return Res;
}

void print_sparse(sparse_matrix* A){
	int i,j,n;
	double sum;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/sparse_matrix","w");
		for (i=0;i<A->size;i++){
			n = A->Len[i]-1;
			sum = A->Values[i][n];
			for (j=0;j<n;j++){
				sum += A->Values[i][j];
				fprintf(file,"(%d|%e)  ",A->Indices[i][j],A->Values[i][j]);
			}
			fprintf(file,"(%d|%e)\n",A->Indices[i][n],A->Values[i][n]);
		}		
		fclose(file);
	}
	else{printf("Matrix = NULL");}
}

void print_3D_sparse(sparse_matrix3D* A){
	int i,j,n;
	double sum;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/sparse_matrix3D","w");
		for (i=0;i<A->size;i++){
			n = A->Len[i]-1;
			sum = A->Values[i][n];
			for (j=0;j<n;j++){
				sum += A->Values[i][j];
				fprintf(file,"(%d,%d|%f)  ",A->Indices1[i][j],A->Indices2[i][j],A->Values[i][j]);
			}
			fprintf(file,"(%d,%d|%f)\n",A->Indices1[i][n],A->Indices2[i][n],A->Values[i][n]);
		}		
		fclose(file);
	}
	else{printf("Matrix = NULL");}
}

void print_matrix_diff(char* Name1,char* Name2,double tol){
	int i,j,m;
	char* Path = "/Home/damage/radszuwe/Daten";
	sparse_matrix* A = read_sparse_textfile(Path,Name1);
	sparse_matrix* B = read_sparse_textfile(Path,Name2);
	if (A->size != B->size){
		printf("matrices have different size: size1=%d size2=%d\n",A->size,B->size);
	}
	else{
		sparse_add(A,B,-1.);
		for (i=0;i<A->size;i++){
			m = A->Len[i];
			j = 0;
			while(j<m){
				if (fabs(A->Values[i][j])<tol && A->Indices[i][j]!=i){
					remove_element_at(A,i,j);
					m = A->Len[i];				
				}
				else j++;
			}
		}
		print_sparse(A);
	}
	free_sparse(A);
	free_sparse(B);
}

void write_sparse_textfile(char* path,char* name,sparse_matrix* A){
	int i,j,n;
	double sum;
	if (A!=NULL){
		char filename[512];
		sprintf(filename,"%s/%s",path,name);
		FILE* file = fopen(filename,"w");
		for (i=0;i<A->size;i++){
			n = A->Len[i]-1;
			sum = A->Values[i][n];
			for (j=0;j<n;j++){
				sum += A->Values[i][j];
				fprintf(file,"(%d|%f)  ",A->Indices[i][j],A->Values[i][j]);
			}
			fprintf(file,"(%d|%f)\n",A->Indices[i][n],A->Values[i][n]);
		}		
		fclose(file);
	}
	else{printf("Matrix = NULL");}
}

void print_sparse_table(sparse_matrix* A){
	int i,j,n;
	double* R;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/sparse_table","w");
		for (i=0;i<A->size;i++){
			n = A->Len[i]-1;
			R = zero_vector(A->size);
			for (j=0;j<=n;j++) R[A->Indices[i][j]] = A->Values[i][j];
			for (j=0;j<A->size-1;j++) fprintf(file,"%f ",R[j]);
			fprintf(file,"%f\n",R[A->size-1]);
			free(R);
		}
		fclose(file);
	}
	else{printf("Matrix = NULL");}
}

void print_vector(double* V,int n){
	int i;
	if (V!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/vector","w");
		for (i=0;i<n;i++){
			fprintf(file,"%f\n",V[i]);
		}
		fclose(file);
	}
}

void print_scalar_data(char* Name,double* V,int n){
	int i;
	if (V!=NULL){
		FILE* file = fopen(Name,"w");
		if (file!=NULL){
			for (i=0;i<n;i++){
				fprintf(file,"%e\n",V[i]);
			}
		fclose(file);
		}
		else printf("could not write to file %s\n -> skip",Name);
	}
}

void print_table(double* V,char* name,int n){
	int i;
	char dir[512];
	sprintf(dir,"/Home/damage/radszuwe/Daten/%s",name);
	if (V!=NULL){
		FILE* file = fopen(dir,"w");
		for (i=0;i<n;i++){
			fprintf(file,"%e\n",V[i]);
		}
		fclose(file);
	}
}

void print_list(int* V,int n){
	int i;
	if (V!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/list","w");
		for (i=0;i<n;i++){
			fprintf(file,"(%d|%d)\n",i,V[i]);
		}
		fclose(file);
	}
}

int test_sparse_matrix(sparse_matrix* A){
	int i,j,k;	
	int n = A->size;
	int res = 1;
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			if (k>=n || k<0){
				res = 0;
				printf("Fehler bei Index %d:%d Index=%d\n",i,j,k);
				break;
			}
		}
	}	
	return res;
}

int insert_sparse(sparse_matrix* A,double a,int i,int j){
	int flag = 0;
	int pos = get_position(A->Indices[i],A->Len[i],j,&flag);
	if (flag==-1){insert_element_at(A,a,i,j,pos);}
	else{
		A->Values[i][pos] += a;
		if (A->Indices[i][pos]!=j){printf("Fehler bei Zeile %d\n",i);}
	}
	return pos;
}

void set_sparse(sparse_matrix* A,double a,int i,int j){
	int flag = 0;
	int pos = get_position(A->Indices[i],A->Len[i],j,&flag);
	if (flag==-1){insert_element_at(A,a,i,j,pos);}
	else{
		A->Values[i][pos] = a;
		if (A->Indices[i][pos]!=j){printf("Fehler bei Zeile %d\n",i);}
	}
}

void insert_element_at(sparse_matrix* A,double a,int i,int j,int pos){
	int k;
	int oldsize = A->Len[i];
	double* NewVal = (double*)realloc(A->Values[i],(oldsize+1)*sizeof(double));
	int* NewInd = (int*)realloc(A->Indices[i],(oldsize+1)*sizeof(int));
	if (NewVal!=NULL) A->Values[i] = NewVal;
	if (NewInd!=NULL) A->Indices[i] = NewInd;
	for (k=oldsize-1;k>=pos;k--){
		A->Values[i][k+1] = A->Values[i][k];
		A->Indices[i][k+1] = A->Indices[i][k];
	}
	A->Values[i][pos] = a;
	A->Indices[i][pos] = j;
	A->Len[i]++;
}

void remove_element_at(sparse_matrix* A,int i,int pos){
	int k;
	int oldsize = A->Len[i];
	for (k=pos;k<oldsize-1;k++){
		A->Values[i][k] = A->Values[i][k+1];
		A->Indices[i][k] = A->Indices[i][k+1];
	}
	if (oldsize>1){
		A->Values[i] = (double*)realloc(A->Values[i],(oldsize-1)*sizeof(double));
		A->Indices[i] = (int*)realloc(A->Indices[i],(oldsize-1)*sizeof(int));
		A->Len[i]--;
	}
	else{
		A->Values[i] = (double*)realloc(A->Values[i],sizeof(double));
		A->Indices[i] = (int*)realloc(A->Indices[i],sizeof(int));
		A->Values[i][0] = 0;
		A->Indices[i][0] = i;
	}
}

void remove_element(sparse_matrix* A,int i,int j){
	int flag = 0;
	int pos = get_position(A->Indices[i],A->Len[i],j,&flag);
	if (flag>=0) remove_element_at(A,i,pos);
}

int find_element(sparse_matrix* A,int j){
	int i,k;
	for (i=0;i<A->size;i++){
		for (k=0;k<A->Len[i];k++){
			if (A->Indices[i][k]==j){return i;}
		}
	}
	return -1;
}

void sparse_in_sparse(sparse_matrix* A,sparse_matrix* B,int i_o,int j_o){
	int i,j,k;
	double b;
	for (i=0;i<B->size;i++){
		for (k=0;k<B->Len[i];k++){
			j = B->Indices[i][k];
			b = B->Values[i][k];
			if (b!=0) set_sparse(A,b,i+i_o,j+j_o);
		}
	}
}

int* get_nonzero_indices(double* Vector,int size,int* new_size){
	int i;
	int* Res = NULL;
	int k = 0;
	for (i=0;i<size;i++){
		if (Vector[i]!=0){
			k++;
			Res = (int*)realloc(Res,k*sizeof(int));
			Res[k-1] = i;
		}
	}
	*new_size = k;
	return Res;
}

double get_matrix_element(sparse_matrix* A,int i,int j){
	int flag = 0;
	int pos = get_position(A->Indices[i],A->Len[i],j,&flag);
	//printf("flag bei %d:%d\t %d\n",i,j,flag);
	double res = 0;
	if (flag==1){res = A->Values[i][pos];}
	return res;
}

void reset_row(sparse_matrix* A,int i){
	free(A->Values[i]);
	free(A->Indices[i]);
	A->Values[i] = (double*)malloc(sizeof(double));
	A->Values[i][0] = 0;
	A->Indices[i] = (int*)malloc(sizeof(int));
	A->Indices[i][0] = i;
	A->Len[i] = 1;
}

void insert_row(sparse_matrix* A,double* R,int i){
	int j;
	reset_row(A,i);
	for (j=0;j<A->size;j++) if (R[j]!=0){
		insert_sparse(A,R[j],i,j);
	}
}

sparse_matrix* cols_to_sparse(double** V,int col_num,int v_size){
	int i,j;
	sparse_matrix* Res = sparse_zero(v_size);
	for (i=0;i<col_num;i++){
		for (j=0;j<v_size;j++) if (V[i][j]!=0) insert_sparse(Res,V[i][j],j,i);
	}
	return Res;
}

sparse_matrix* rows_to_sparse(double** V,int row_num,int v_size){
	int i,j;
	sparse_matrix* Res = sparse_zero(row_num);
	for (i=0;i<row_num;i++){
		for (j=0;j<v_size;j++) if (V[i][j]!=0) insert_sparse(Res,V[i][j],i,j);
	}
	return Res;
}

int get_max_col_index(sparse_matrix* A){
	int i,j;
	int max = A->Indices[0][0];
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++) if (A->Indices[i][j]>max) max = A->Indices[i][j];
	}
	return max;
}

sparse_matrix* get_transpose(sparse_matrix* A,int new_size){
	int i,j,r;
	int n = A->size;
	sparse_matrix* Res = sparse_zero(new_size);
	for (i=0;i<n;i++){
		for (r=0;r<A->Len[i];r++){
			j = A->Indices[i][r];
			insert_sparse(Res,A->Values[i][r],j,i);
		}
	}
	return Res;
}

int is_symmetric(sparse_matrix* A,int i,int j){
	int k;
	int res = 0;
	int J = A->Indices[i][j];
	double a = A->Values[i][j];
	if (J<A->size) for (k=0;k<A->Len[J];k++){
		if (A->Indices[J][k]==i && A->Values[J][k]==a){
			res = 1;
			break;
		}
	}
	return res;
}

int symmetry(sparse_matrix* A,double tol){
	int i,j,k,ind,match;
	double val;
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++){
			ind = A->Indices[i][j];
			val = A->Values[i][j];
			match = 0;
			for (k=0;k<A->Len[ind];k++){
				if (A->Indices[ind][k]==i && fabs(A->Values[ind][k]-val)<tol) match = 1;
			}
			if (match==0 && fabs(val)>tol){
				printf("asymmetric index found at (%d|%d)\n",i,ind);
				return 0;
			}
		}
		
	}
	return 1;
}

void sparse_approximate(sparse_matrix* A,double tol){
	int i,j,k;
	double a,max;
	int counter = 0;
	int total = 0;
	int n = A->size;
	for (i=0;i<n;i++){
		max = 0;
		total +=A->Len[i];
		for (j=0;j<A->Len[i];j++){
			a = fabs(A->Values[i][j]);
			if (A->Indices[i][j]!= i && a>max) max = a;
		}
		j = 0;
		while (j<A->Len[i]){
			k = A->Indices[i][j];
			if (k!=i && fabs(A->Values[i][j])<max*tol){
				remove_element_at(A,i,j);
				counter++;
			}
			else j++;
		}
	}
	printf("sparse approximate: fraction %e of elements removed\n",(double)counter/total);
}

double degree_of_occupation(sparse_matrix* A){
	int i;
	int n = A->size;
	double sum = 0;
	for (i=0;i<n;i++) sum += (double)A->Len[i];
	sum /= (double)n*n;
	return sqrt(sum);
}

sparse_matrix* clone(sparse_matrix* A){
	int i,j;
	int n = A->size;
	sparse_matrix* Res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	Res->size = n;
	Res->Len = (int*)malloc(n*sizeof(int));
	Res->Values = (double**)malloc(n*sizeof(double*));
	Res->Indices = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++){
		Res->Len[i] = A->Len[i];
		Res->Values[i] = (double*)malloc((A->Len[i])*sizeof(double));
		Res->Indices[i] = (int*)malloc((A->Len[i])*sizeof(int));
		for (j=0;j<A->Len[i];j++){
			Res->Values[i][j] = A->Values[i][j];
			Res->Indices[i][j] = A->Indices[i][j];
		}
	}
	return Res;
}

void copy_sparse_content(sparse_matrix* A,sparse_matrix* B){		//    copy B -> A
	int i,j,m;
	int n = B->size;
	for (i=0;i<A->size;i++){
		free(A->Values[i]);
		free(A->Indices[i]);
	}
	A->size = n;
	A->Len = (int*)realloc(A->Len,n*sizeof(int));
	A->Values = (double**)realloc(A->Values,n*sizeof(double*));
	A->Indices = (int**)realloc(A->Indices,n*sizeof(int*));
	for (i=0;i<n;i++){
		m = B->Len[i];
		A->Len[i] = m;
		A->Values[i] = (double*)malloc(m*sizeof(double));
		A->Indices[i] = (int*)malloc(m*sizeof(int));
		for (j=0;j<m;j++){
			A->Values[i][j] = B->Values[i][j];
			A->Indices[i][j] = B->Indices[i][j];
		}
	}
}

int rec_get_position(int* list,int val,int l,int r,int* Flag){
	int pos = (l+r)/2;
	if (pos == l){
		*Flag = -1;
		return pos+1;
	}
	else{
		if (val>list[pos]) return rec_get_position(list,val,pos,r,Flag);
		if (val<list[pos]) return rec_get_position(list,val,l,pos,Flag);
		if (val==list[pos]){
			*Flag = 1;
			return pos;
		}
	}
	return -1;
}

int get_position(int* list,int size,int val,int* Flag){  // list muss sortiert sein !
	*Flag = -1;
	if (size>0){
		if (val>list[0] && val<list[size-1]){return rec_get_position(list,val,0,size-1,Flag);}
		else{
			if (val<=list[0]){
				if (val==list[0]){*Flag = 1;}else{*Flag = -1;}
				return 0;
			}
			if (val>=list[size-1]){
				if (val==list[size-1]){
					*Flag = 1;
					return size-1;
				}
				else{
					*Flag = -1;
					return size;
				}
			}
		}
	}
	return -1;
}

void double_sort(int* List,double* Values,int size){  // divide & conquer
	if (size>1){									  // List wird permutiert
		int i;										  // Values bleibt unverändert
		int s_size = 0;
		int b_size = 0;
		int a = List[0];
		int* Smaller = (int*)malloc(s_size*sizeof(int));
		int* Bigger = (int*)malloc(b_size*sizeof(int));
		for (i=1;i<size;i++) if (Values[List[i]]>Values[a]){
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
		double_sort(Smaller,Values,s_size);
		double_sort(Bigger,Values,b_size);
		for (i=0;i<s_size;i++) List[i] = Smaller[i];
		List[s_size] = a;
		for (i=0;i<b_size;i++) List[s_size+i+1] = Bigger[i];
		free(Smaller);
		free(Bigger);
	}
}

double row_times_row(sparse_matrix* A,sparse_matrix* B,int i_a,int i_b,int j_min,int j_max){
	int l = 0;
	int k = A->Indices[i_a][0];
	int n_a = A->Len[i_a];
	int n_b = B->Len[i_b];
	while (k<j_min){
		l++;
		if (l<n_a){k = A->Indices[i_a][l];}else{break;}
	}
	int s = 0;
	int r = B->Indices[i_b][s];
	double sum = 0;
	if(l<n_a){
		while (k<=j_max){
			while (r<k){
				s++;
				if (s<n_b){r = B->Indices[i_b][s];}else{break;}
			}
			if (r==k){sum += A->Values[i_a][l]*B->Values[i_b][s];}
			l++;
			if (l<n_a){k = A->Indices[i_a][l];}else{break;}
		}
	}
	return sum;
}

int in_list(int* List,int size,int n){
	int i;
	for (i=0;i<size;i++){
		if(List[i]==n){return i;}
	}
	return -1;
}

double** create_2D_array(int n,int m){
	int i,j;
	double** Res = (double**)malloc(n*sizeof(double*));
	for (i=0;i<n;i++){
		Res[i] = (double*)malloc(m*sizeof(double));
		for (j=0;j<m;j++) Res[i][j] = 0; 
	}
	return Res;
}

void print_sqr_array(double** A,int n,int m){
	int i,j;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=0;j<m;j++){
				fprintf(file,"%e   ",A[j][i]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
}

void print_2D_array(double** A,int n){
	int i,j;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=1;j<=A[i][0];j++){
				fprintf(file,"%e   ",A[i][j]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
}

void print_2D_list(int** A,int n){
	int i,j;
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=1;j<=A[i][0];j++){
				fprintf(file,"%d   ",A[i][j]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
}

int test_vector(double* X,int n,char* message){
	int i;
	for (i=0;i<n;i++) if (isnan(X[i])){
		printf("%s\n",message);
		fflush(stdout);
		return 1;
	}
	return 0;
}

int test_negative(double* X,int n){
	int i;
	for (i=0;i<n;i++) if (X[i]<0){
		printf("negative component at index %d\n",i);
		return 1;
	}
	return 0;
}

void remove_negative(double* X,int n){
	int i;
	for (i=0;i<n;i++) if (X[i]<0) X[i] = 0;
}

void test_ILU(double** A,double** L,double** U,int n){
	int i,j,k;
	double sum = 0;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			if (A[i][j]!=0){
				if (j<=i){
					sum = A[i][j];
					if (i==1 && j==1){
						printf("hier");
					}
					for (k=0;k<j;k++){
						sum -= L[i][k]*U[k][j];
					}
					L[i][j] = sum;
					U[i][j] = 0;
				}
				else{
					sum = A[i][j];
					for (k=0;k<i;k++){sum -= L[i][k]*U[k][j];}
					sum /= L[i][i];
					L[i][j] = 0;
					U[i][j] = sum;
				}
			}
		}
		U[i][i] = 1;
	}
}

int incomplete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	double sum,d;
	int i,j,jr;
	int n = A->size;
	for (i=0;i<n;i++){
		insert_sparse(U,1.,i,i);
		for (jr=0;jr<A->Len[i];jr++){
			j = A->Indices[i][jr];
			if (j<=i){
				sum = A->Values[i][jr];
				sum -= row_times_row(L,U,i,j,0,j-1);
				insert_sparse(L,sum,i,j);
			}
			else{
				sum = A->Values[i][jr];
				sum -= row_times_row(L,U,i,j,0,i-1);
				d = get_matrix_element(L,i,i);
				if (d!=0) sum /= d;
				else{
					//printf("ILU: Warning, diagonal element is zero at index %d!\n",i);
					return 0;
				}
				insert_sparse(U,sum,j,i);
			}
		}
	}
	sparse_matrix* TrU = get_transpose(U,n);
	copy_sparse_content(U,TrU);
	free_sparse(TrU);
	return 1;
}

void diagonal_preconditioner(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	int i;
	double a;
	int n = A->size;
	for (i=0;i<n;i++){
		a = get_matrix_element(A,i,i);
		if (a==0){
			printf("Diagonalelement in Zeile %d ist null\n",i);
			break;
		}
		insert_sparse(L,a,i,i);
		insert_sparse(U,1,i,i);
	}
}

double residuum(void (*Matvec)(int n,double* y,double* z),double* Sol,double* b,int n){
	double* R = zero_vector(n);
	(*Matvec)(n,Sol,R);
	vector_add(R,b,-1,n);
	double r = euklid_norm(R,n);
	free(R);
	return r;
}

void L_triang_invert(sparse_matrix* L,double* x){
	int i,j,J;
	double sum;
	int n = L->size;
	double* b = clone_vector(x,n);
	double diag = 0;
	for (i=0;i<n;i++){
		sum = b[i];
		for (j=0;j<L->Len[i];j++){
			J = L->Indices[i][j];
			if (J!=i) sum -= L->Values[i][j]*x[J];
			else diag = L->Values[i][j];
		}
		x[i] = sum/diag;
	}
	free(b);
}

void U_triang_invert(sparse_matrix* U,double* x){
	int i,j,J;
	double diag = 0;
	int n = U->size;
	double sum = 0;
	double* b = clone_vector(x,n);
	
	for (i=n-1;i>=0;i--){
		sum = b[i];
		for (j=0;j<U->Len[i];j++){
			J = U->Indices[i][j];
			if (J!=i) sum -= U->Values[i][j]*x[J];
			else diag = U->Values[i][j];
		}
		x[i] = sum/diag;
	}
	free(b);
}

void partial_L_invert(sparse_matrix* LT,double* X,double* Sum,int i,int s){
	int j,k,len;
	for (j=i;j<i+s;j++){
		len = LT->Len[j];
		X[j] = (X[j]-Sum[j])/(LT->Values[j][0]);
		k = 1;
		while(k<len && LT->Indices[j][k]<(i+s)){
			Sum[LT->Indices[j][k]] += (LT->Values[j][k])*X[j];
			k++;
		}
	}
}

void partial_U_invert(sparse_matrix* UT,double* X,double* Sum,int i,int s){
	int j,k,len;
	for (j=i;j>i-s;j--){
		len = UT->Len[j];
		X[j] = (X[j]-Sum[j])/(UT->Values[j][len-1]);
		k = len-2;
		while(k>=0 && UT->Indices[j][k]>(i-s)){
			Sum[UT->Indices[j][k]] += (UT->Values[j][k])*X[j];
			k--;
		}
	}
}

int* ppartial_L_invert(sparse_matrix* LT,double* X,double* Sum,int i,int s){
	int j,k,len;
	int* Res = zero_int_list(s);
	for (j=i;j<i+s;j++){
		len = LT->Len[j];
		X[j] = (X[j]-Sum[j])/(LT->Values[j][0]);
		k = 1;
		while(k<len && LT->Indices[j][k]<(i+s)){
			Sum[LT->Indices[j][k]] += (LT->Values[j][k])*X[j];
			k++;
		}
		Res[j-i] = k;
	}
	return Res;
}

int* ppartial_U_invert(sparse_matrix* UT,double* X,double* Sum,int i,int s){
	int j,k,len;
	int* Res = zero_int_list(s);
	for (j=i;j>i-s;j--){
		len = UT->Len[j];
		X[j] = (X[j]-Sum[j])/(UT->Values[j][len-1]);
		k = len-2;
		while(k>=0 && UT->Indices[j][k]>(i-s)){
			Sum[UT->Indices[j][k]] += (UT->Values[j][k])*X[j];
			k--;
		}
		Res[i-j] = k;
	}
	return Res;
}

void pL_triang_invert(sparse_matrix* LT,double* X,int s){
	int i,j,k,len;
	int* Dest;
	//int* Ind;
	double x,y;
	//double* Val;
	int n = LT->size;
	double* Sum = zero_vector(n);
	for (i=0;i<n;i+=s){
		Dest = ppartial_L_invert(LT,X,Sum,i,min(n-i,s));    // seriell
#line 1457 "linear_algebra.c"
		#pragma omp parallel private(j,k,len,x,y) shared(Sum)
		{
#line 1459 "linear_algebra.c"
			#pragma omp for schedule(dynamic,2)
			for (j=i;j<min(i+s,n);j++){						// parallel
				len = LT->Len[j];
				x = X[j];
				/*Val = &(LT->Values[j][len-1]);
				Ind = &(LT->Indices[j][len-1]);
				for (k=len-1;k>=Dest[j-i];k--){
					Sum[*Ind] += (*Val)*x;
				}*/
				for (k=len-1;k>=Dest[j-i];k--){
					y = Sum[LT->Indices[j][k]]+LT->Values[j][k]*x;
#line 1470 "linear_algebra.c"
					#pragma omp critical
					Sum[LT->Indices[j][k]]=y;
				}
			}
		}
		free(Dest);
	}
	free(Sum);
}

void pU_triang_invert(sparse_matrix* UT,double* X,int s){
	int i,j,k,len;
	int n = UT->size;
	int* Dest;
	//int* Ind;
	double x,y;
	//double* Val;
	double* Sum = zero_vector(n);
	for (i=n-1;i>=0;i-=s){
		Dest = ppartial_U_invert(UT,X,Sum,i,min(i+1,s));    // seriell
#line 1490 "linear_algebra.c"
		#pragma omp parallel private(j,k,len,x,y) shared(Sum) 
		{
#line 1492 "linear_algebra.c"
			#pragma omp for schedule(dynamic,2)
			for (j=i;j>max(i-s,-1);j--){				// parallel
				len = UT->Len[j];
				x = X[j];
				/*Val = UT->Values[j];
				Ind = UT->Indices[j];
				for (k=0;k<=Dest[i-j];k++){
					Sum[*Ind] += (*Val)*x;
					Val++;
					Ind++;
				}*/
				for (k=0;k<=Dest[i-j];k++){
					y = Sum[UT->Indices[j][k]]+UT->Values[j][k]*x;
#line 1505 "linear_algebra.c"
					#pragma omp critical
					Sum[UT->Indices[j][k]]=y;
				}
			}
		}
		free(Dest);
	}
	free(Sum);
}

sparse_matrix* get_inv_U_triang(sparse_matrix* U){
	int i,j;
	int n = U->size;
	sparse_matrix* Res = sparse_zero(n);
	for (i=0;i<n;i++){
		double* E = zero_vector(n);
		E[i] = 1.;
		U_triang_invert(U,E);
		for(j=0;j<n;j++) if (E[i]!=0) insert_sparse(Res,E[j],j,i);
		free(E);
	}
	return Res;
}

double* get_inv_diag(sparse_matrix* A){
	int i;
	double d;
	double* Res = zero_vector(A->size);
	for (i=0;i<A->size;i++){
		d = get_diag(A,i);
		if (d!=0) Res[i] = 1./d; 
		else{
			printf("zero diagonal at index %d -> abort\n",i);
			return NULL;
		}
	}
	return Res;
}



void FTI_test(sparse_matrix* L,sparse_matrix* U){
	
	
	/* int n = 4;
	sparse_matrix* L = sparse_identity(n);
	insert_sparse(L,0.10,1,0);
	insert_sparse(L,0.20,2,0);
	insert_sparse(L,0.30,3,0);
	insert_sparse(L,0.21,2,1);
	insert_sparse(L,0.31,3,1);
	insert_sparse(L,0.32,3,2);
	
	sparse_matrix* U = sparse_zero(n);
	insert_sparse(U,3.,0,0);
	insert_sparse(U,5.,1,1);
	insert_sparse(U,-2.,2,2);
	insert_sparse(U,6.,3,3);
	insert_sparse(U,0.00001,0,1);
	insert_sparse(U,0.20,0,2);
	insert_sparse(U,0.30,0,3);
	insert_sparse(U,0.21,1,2);
	insert_sparse(U,0.31,1,3);
	insert_sparse(U,0.32,2,3);
	
	double* X = zero_vector(n);
	X[0] = 1;
	X[1] = 2;
	X[2] = 3;
	X[3] = 4;*/
	
	/*int n = L->size;
	sparse_matrix* LT = get_transpose(L);
	sparse_matrix* UT = get_transpose(U);
	
	double* X = zero_vector(n);
	int i,j;
	for (i=0;i<n;i++) X[i] = sin((double)1000.*i);
	double* Y = clone_vector(X,n);
	int N = 100;
	
	#ifdef _OPENMP
	omp_set_num_threads(2);
	double start = omp_get_wtime();
	#endif
	printf("start computation\n");
	fflush(stdout);
	for (j=0;j<N;j++){
		linear_map(X,0.00001,L,X);
	}
	#ifdef _OPENMP
	printf("parallel time: %f sec\n",(double)(omp_get_wtime()-start));
	fflush(stdout);
	start = omp_get_wtime();
	#endif
	
	printf("start computation\n");
	fflush(stdout);
	for (j=0;j<N;j++){
		linear_map(Y,0.00001,L,Y);
	}
	#ifdef _OPENMP
	printf("seriel time: %f sec\n",(double)(omp_get_wtime()-start));
	fflush(stdout);
	#endif
	printf("abweichung: %f\n",vec_dist(X,Y,n)/euklid_norm(Y,n));
	
	exit(0);*/
}


int get_max_band_width(sparse_matrix* A){
	int i,d;
	int max = 0;
	for (i=0;i<A->size;i++){
		if (A->Len[i]>0) d = A->Indices[i][A->Len[i]-1]-A->Indices[i][0]+1; else d = 0;
		if (d>max) max = d;
	}
	return max;
}

int get_max_mem_bandwidth(sparse_matrix* A){
	int i;
	int max = 0;
	if (A!=NULL){
		for (i=0;i<A->size;i++){
			if (A->Len[i]>max) max = A->Len[i];
		}
	}
	return max;
}

int get_ave_band_width(sparse_matrix* A){
	int i,w;
	int n = 0;
	int sum = 0;
	for (i=0;i<A->size;i++){
		if (A->Len[i]>0){
			w = A->Indices[i][A->Len[i]-1]-A->Indices[i][0]+1;
			sum += w;
			if (w<0) printf("warning: wrong index ordering in line %d\n",i);
			n++;
		}
	}
	return ceil((double)sum/n);
}

int L_triang_fast_invert(sparse_matrix* L,int** Ind,double** Val,int size,int I,int max_band_width,double tol){
	int i,j,k,K,next,last;
	double a,sum,norm;
	double diag = 0;
	int n = L->size;
	double* X = zero_vector(n);
	for (i=0;i<size;i++){
		j = Ind[I][i];
		norm = 0;
		last = (j-max_band_width<0)?0:j-max_band_width;
		for (k=last;k<j;k++) norm +=fabs(X[k]);
		sum = Val[I][i];
		if (i<size-1) next = Ind[I][i+1]; else next = n;
		while(j<next){
			for (k=0;k<L->Len[j];k++){
				K = L->Indices[j][k];
				a = L->Values[j][k];
				if (K!=j){
					sum -= a*X[K];
					norm += fabs(X[K]);
				}
				else diag = a;
			}
			X[j] = sum/diag;
			last = j-max_band_width;
			norm += fabs(X[j]);
			if (last>=0) norm -= fabs(X[last]);
			if (fabs(diag)*norm<tol) break;
			j++;
			sum = 0;
		}
	}
	int new_size = 0;
	for (i=0;i<n;i++) if (X[i]!=0){
		Ind[I] = (int*)realloc(Ind[I],(new_size+1)*sizeof(int));
		Val[I] = (double*)realloc(Val[I],(new_size+1)*sizeof(double));
		Ind[I][new_size] = i;
		Val[I][new_size] = X[i];
		new_size++;
	}
	free(X);
	return new_size;
}

int U_triang_fast_invert(sparse_matrix* U,int**Ind,double** Val,int size,int I,int max_band_width,double tol){
	int i,j,k,K,next,last;
	double a,sum,norm;
	double diag = 0;
	int n = U->size;
	double* X = zero_vector(n);
	for (i=size-1;i>=0;i--){
		j = Ind[I][i];
		sum = Val[I][i];
		norm = 0;
		last = (j+max_band_width<n)?j+max_band_width:n-1;
		for (k=last;k>j;k--) norm += fabs(X[k]);
		if (i>0) next = Ind[I][i-1]; else next = -1;
		while(j>next){
			for (k=0;k<U->Len[j];k++){
				K = U->Indices[j][k];
				a = U->Values[j][k];
				if (K!=j){
					sum -= a*X[K];
					norm += fabs(X[K]);
				}
				else diag = a;
			}
			X[j] = sum/diag;
			last = j+max_band_width;
			norm += fabs(X[j]);
			if (last<n) norm -= fabs(X[last]);
			if (fabs(diag)*norm<tol) break;
			j--;
			sum = 0;
		}
	}
	int new_size = 0;
	for (i=0;i<n;i++) if (X[i]!=0.){
		Ind[I] = (int*)realloc(Ind[I],(new_size+1)*sizeof(int));
		Val[I] = (double*)realloc(Val[I],(new_size+1)*sizeof(double));
		Ind[I][new_size] = i;
		Val[I][new_size] = X[i];
		new_size++;
	}
	free(X);
	return new_size;
}

sparse_matrix* get_fast_ILU_preconditioned_matrix(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U,double tol){
	int i,L_width,U_width;
	int n = A->size;
	L_width = get_max_band_width(L);
	U_width = get_max_band_width(L);
	printf("max. bandwidth L: %d\n",L_width);
	printf("max. bandwidth U: %d\n",U_width);
	sparse_matrix* Transpose = get_transpose(A,n);
	for (i=0;i<n;i++) Transpose->Len[i] = L_triang_fast_invert(L,Transpose->Indices,Transpose->Values,Transpose->Len[i],i,L_width,tol);
	sparse_approximate(Transpose,1E-2);
	for (i=0;i<n;i++) Transpose->Len[i] = U_triang_fast_invert(U,Transpose->Indices,Transpose->Values,Transpose->Len[i],i,U_width,tol);
	sparse_approximate(Transpose,1E-4);
	sparse_matrix* Res = get_transpose(Transpose,n);
	free_sparse(Transpose);
	return Res;
}

double get_diag_dominance(sparse_matrix* A){
	int i,j,J;
	double sum;
	double max = 0;
	double diag = 0;
	for (i=0;i<A->size;i++){
		sum = 0;
		for (j=0;j<A->Len[i];j++){
			J = A->Indices[i][j];
			if (J!=i) sum += fabs(A->Values[i][j]);
			else diag = fabs(A->Values[i][j]);
		}
		sum /= diag;
		if (sum>max) max = sum;
	}
	return max;
}

double matrix_residuum(sparse_matrix* A,double* Sol,double* b){
	int n = A->size;
	double* R = clone_vector(b,n);
	linear_map(R,-1.,A,Sol);
	double r = euklid_norm(R,n);
	free(R);
	return r;
}

double ILU_residuum(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U,
 double* Sol,double* b){
	int n = A->size;
	double* R = clone_vector(b,n);
	linear_map(R,-1.,A,Sol);
	L_triang_invert(L,R);
	U_triang_invert(U,R);
	double r = euklid_norm(R,n);
	free(R);
	return r;
}

double SOR_residuum(sparse_matrix* A,double* Sol,double* b,int alpha){
	int n = A->size;
	double* R = clone_vector(b,n);
	linear_map(R,-1.,A,Sol);
	double* P = zero_vector(n);
	SOR(A,R,P,alpha,1);
	double r = euklid_norm(P,n);
	free(R);
	free(P);
	return r;
}

void SOR(sparse_matrix* A,double* F,double* Sol,double alpha,int iter){
	int i,j,k,ind,diag;							//SOR-Verfahren
	double sum;
	int n = A->size;
	for (k=0;k<iter;k++){
		for (i=0;i<n;i++){
			sum = F[i];
			diag = -1;
			for (j=0;j<A->Len[i];j++){
				ind = A->Indices[i][j];
				if (ind!=i) sum -= A->Values[i][j]*Sol[ind];
				else diag = j;
			}
			if (diag<0){
				printf("diagonal element is zero at index %d -> abort SOR\n",i);
				exit(0);
			}
			Sol[i] = (1.-alpha)*Sol[i]+alpha*sum/A->Values[i][diag];
		}
	}
}

int gauss_seidel(int (*Inv_NB_matvec)(int n,double* x,double* b),
void (*NB_matvec)(int n,double* y,double* z),
double* Solution,double* b,int size,int iterations,double tol){
	int i;
	double r,r0;
	if (iterations>0){
		for (i=0;i<iterations;i++){
			if ((*Inv_NB_matvec)(size,Solution,b)<0){
				printf("->Abbruch\n");
				break;
			}
		}
		return iterations;
	}
	else{
		r0 = residuum(NB_matvec,Solution,b,size);
		i = 0;
		do{
			if ((*Inv_NB_matvec)(size,Solution,b)<0){
				printf("->Abbruch\n");
				break;	
			}
			r = residuum(NB_matvec,Solution,b,size);
			i++;
			//printf("i= %d | r= %f\n",i,r/r0);
		}while(r/r0>tol && i<-iterations);
		//if (i>=-iterations){printf("Warnung: Maximale Anzahl an GIterationen erreicht !\n");}
		//printf(" %d Iterationen Residuum %f\n",i,r/r0);
		return i;
	}
}

void exchange_row(sparse_matrix* A,int n,int m){  // Achtung memory leak ?
	int buffer_len = A->Len[m];
	int* Buffer_ind = clone_list(A->Indices[m],A->Len[m]);
	double* Buffer_val = clone_vector(A->Values[m],A->Len[m]);
	A->Indices[m] = (int*)realloc(A->Indices[n],(A->Len[n])*sizeof(int));
	A->Values[m] = (double*)realloc(A->Values[n],(A->Len[n])*sizeof(double));
	A->Len[m] = A->Len[n];
	A->Indices[n] = (int*)realloc(Buffer_ind,buffer_len*sizeof(int));
	A->Values[n] = (double*)realloc(Buffer_val,buffer_len*sizeof(double));
	A->Len[n] = buffer_len;
	//free(Buffer_ind);
	//free(Buffer_val);
}

void pivot(sparse_matrix* A,double* B,int* Map){  // relative Spaltenmaximumsstrategie
	int i,j,k,max_i;
	double diag_i,diag_j,sum,b;
	int n = A->size;
	if (Map!=NULL) for (i=0;i<n;i++) Map[i] = i;
	for (i=0;i<n;i++){
		sum = 0;
		diag_i = fabs(get_diag(A,i));
		for (j=0;j<A->Len[i];j++) if (A->Indices[i][j]>=i) sum += fabs(A->Values[i][j]);
		diag_i /= sum;
		max_i = i;
		for (j=i+1;j<n;j++){
			diag_j = fabs(get_matrix_element(A,j,i));
			if (diag_j>0){
				sum = 0;
				for (k=0;k<A->Len[j];k++) if (A->Indices[j][k]>=i) sum += fabs(A->Values[j][k]);
				diag_j /= sum;
				if (diag_j>diag_i){
					diag_i = diag_j;
					max_i = j;
				}
			}
		}
		if (max_i!=i){
			exchange_row(A,i,max_i);
			b = B[i];
			B[i] = B[max_i];
			B[max_i] = b;
			if (Map!=NULL){
				Map[i] = max_i;
				Map[max_i] = i;
			}
		}
	}
}

int diag_zero(sparse_matrix* A){
	int i;
	for (i=0;i<A->size;i++) if (get_diag(A,i)==0){
		printf("zero diagonal at index %d\n",i);
		return 1;
	}
	return 0;
}

void find_diag_zero(sparse_matrix* A){
	int i;
	for (i=0;i<A->size;i++) if (get_diag(A,i)==0){
		printf("zero diagonal at index %d\n",i);
	}
}

sparse_matrix* make_positive_diag(sparse_matrix* A){
	int i,j;
	sparse_matrix* Res = sparse_identity(A->size);
	for (i=0;i<A->size;i++) if (get_diag(A,i)<0){
		for (j=0;j<A->Len[i];j++) A->Values[i][j] *= -1.;
		Res->Values[i][0] *= -1.;
	}
	return Res;
}

int* get_ID_index_map(int size){
	int i;
	int* Res = zero_int_list(size);
	for (i=0;i<size;i++) Res[i] = i;
	return Res;
}

int* get_inverse_index_map(int* Map,int size){
	int i;
	int* Res = (int*)malloc(size*sizeof(int));
	for (i=0;i<size;i++) Res[i] = -1;
	for (i=0;i<size;i++){
		Res[Map[i]] = i;
	}
	for (i=0;i<size;i++) if (Res[i]<0){
		printf("Achtung: Index map nicht invertierbar ! index %d\n",i);
	}
	return Res;
}

double get_max_element_abs(double* X,int size){
	int i;
	double max = fabs(X[0]);
	for (i=1;i<size;i++){
		if (fabs(X[i])>max) max = fabs(X[i]);
	}
	return max;
}

double get_min_element_abs(double* X,int size){
	int i;
	double min = fabs(X[0]);
	for (i=1;i<size;i++){
		if (fabs(X[i])<min) min = fabs(X[i]);
	}
	return min;
}

void simple_pivot(sparse_matrix* A,double* B,int* Map){				// Spaltenmaximumsstrategie
	int i,j,max_i,start;
	double diag_i,diag_j,b,new_diag;
	int n = A->size;
	printf("simple pivot\n");
	int* Old_map = zero_int_list(n);
	if (Map!=NULL) for (i=0;i<n;i++){
		Old_map[i] = Map[i];
		Map[i] = i;
	}
	for (i=0;i<n;i++){
		diag_i = fabs(get_diag(A,i));
		max_i = i;
		if (diag_i!=0) start = i+1; else start = 0;
		for (j=start;j<n;j++){
			diag_j = fabs(get_matrix_element(A,j,i));
			if (diag_j>diag_i){
				new_diag = get_matrix_element(A,i,j);
				if (new_diag!=0){
					diag_i = diag_j;
					max_i = j;
				}
			}
		}
		if (max_i!=i){
			exchange_row(A,i,max_i);
			if (B!=NULL){
				b = B[i];
				B[i] = B[max_i];
				B[max_i] = b;
			}
			if (Map!=NULL){
				Map[i] = max_i;
				Map[max_i] = i;
			}
		}
		else{
			if (diag_i==0){
				print_sparse(A);
				printf("scheisen! at index %d\n",i);
			}
		}
		if (i % (n/10) ==0) printf("simple pivot: %d \r",100*i/n);
	}
	int* New_map = zero_int_list(n);
	for (i=0;i<n;i++) New_map[i] = Map[Old_map[i]];
	for (i=0;i<n;i++) Map[i] = New_map[i];
	free(Old_map);
	free(New_map);
	printf("simple pivot: completed \n");
}

void check_second_order(sparse_matrix* A,double* B,int* Map,int* Indices,int size){
	int i,j,k,l,p,q,ind_j,ind_l;
	double a,b,ail,alj,aji;
	for (k=0;k<size;k++){
		a = 0;
		i = Indices[k];
		ind_j = Indices[k];
		ind_l = Indices[k];
		for (p=0;p<A->Len[i];p++){
			l = A->Indices[i][p];
			ail = A->Values[i][p];
			for (q=0;q<A->Len[l];q++){
				j = A->Indices[l][q];
				alj = A->Values[l][q];
				aji = get_matrix_element(A,j,i);
				if (fabs(ail*alj*aji)>a){
					a = fabs(ail*alj*aji);
					ind_j = j;
					ind_l = l;
				}
			}
		}
		printf("max 2nd order product index %d: %f\n",Indices[k],a);
		if (a!=0){
			exchange_row(A,i,ind_j);
			exchange_row(A,ind_j,ind_l);
			if (B!=NULL){
				b = B[i];
				B[i] = B[ind_j];
				//B[ind_j] = b;
				//b = B[ind_j];
				B[ind_j] = B[ind_l];
				B[ind_l] = b;
			}
			if (Map!=NULL){
				q = Map[i];
				Map[i] = Map[ind_j];
				//Map[ind_j] = q;
				//q = Map[ind_j];
				Map[ind_j] = Map[ind_l];
				Map[ind_l] = q;
			}
		}
	}
}



void other_pivot(sparse_matrix* A,double* B,int* Map){
	int i,j,k,max_k;
	double maxi,maxk,aik,aki,diag_i,q,b,quality;
	int n = A->size;
	int bad_events = 0;
	printf("other pivot\n");
	int* Bad_ind = NULL;
	for (i=0;i<n;i++){
		diag_i = fabs(get_diag(A,i));
		maxi = get_max_element_abs(A->Values[i],A->Len[i]);
		quality = diag_i*diag_i/(maxi*maxi);
		max_k = i;
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			if (k!=i){
				aik = A->Values[i][j];
				aki = get_matrix_element(A,k,i);
				maxk = get_max_element_abs(A->Values[k],A->Len[k]);
				q = fabs(aik*aki/(maxi*maxk));
				if (q>quality){
					quality = q;
					max_k = k;
				}
			}
		}
		if (quality>0 && max_k!=i){
			exchange_row(A,i,max_k);
			if (B!=NULL){
				b = B[i];
				B[i] = B[max_k];
				B[max_k] = b;
			}
			if (Map!=NULL){
				k = Map[i];
				Map[i] = Map[max_k];
				Map[max_k] = k;
			}
		}
		if (quality==0){
			//print_sparse(A);
			//printf("scheisen! at index %d -> abort\n",i);
			//exit(0);
			bad_events++;
			Bad_ind = (int*)realloc(Bad_ind,bad_events*sizeof(int));
			Bad_ind[bad_events-1] = i;
		}
		if (i % (n/10) ==0) printf("simple pivot: %d \r",100*i/n);
	}
	printf("other pivot completed bad rows: %d\n",bad_events);
	check_second_order(A,B,Map,Bad_ind,bad_events);
	free(Bad_ind);
	find_diag_zero(A);
}

void manuel_pivot(sparse_matrix* A,double* B,int* Map,int* I,int* J,int ind_size,int block_size){
	int i,j,k,l;
	int n = A->size;
	double b;
	printf("manuel pivot\n");
	if (Map!=NULL) for (k=0;k<n;k++) Map[k] = k;
	for (l=0;l<ind_size;l++){
		i = I[l];
		j = J[l];
		for (k=1;k<block_size;k++){
			exchange_row(A,block_size*i+k,block_size*j+k);
			if (B!=NULL){
				b = B[block_size*i+k];
				B[block_size*i+k] = B[block_size*j+k];
				B[block_size*j+k] = b;
			}
			Map[block_size*i+k] = block_size*j+k;
			Map[block_size*j+k] = block_size*i+k;
		}
	}
}

void sparse_row_permutation(sparse_matrix* A,int* Map){
	int i;
	int n = A->size;
	int* len_Buffer = (int*)malloc(n*sizeof(int));
	int** ind_Buffer = (int**)malloc(n*sizeof(int*));
	double** val_Buffer = (double**)malloc(n*sizeof(double*));
	for (i=0;i<n;i++){
		ind_Buffer[i] = A->Indices[i];
		val_Buffer[i] = A->Values[i];
		len_Buffer[i] = A->Len[i];
	}
	for (i=0;i<n;i++){
		A->Indices[i] = ind_Buffer[Map[i]];
		A->Values[i] = val_Buffer[Map[i]];
		A->Len[i] = len_Buffer[Map[i]];
	}
	free(ind_Buffer);
	free(val_Buffer);
	free(len_Buffer);
}

void vector_permutation(double* F,int* Map,int n){
	int i;
	double* val_Buffer = clone_vector(F,n);
	for (i=0;i<n;i++) F[i] = val_Buffer[Map[i]];
	free(val_Buffer);
}


double* Lower_triangular_invert(double** L,double* X,int size){  // solves LX=Y
	int i,j;
	double* Y = clone_vector(X,size);
	for (i=0;i<size;i++){
		//Y[i] = X[i];
		for (j=0;j<i;j++) Y[i] -= L[i][j]*Y[j];
		if (L[i][i]==0){
			printf("L inversion not possible: zero diagonal at index %d\n",i);
			exit(0);
		}
		Y[i] /= L[i][i];
	}
	return Y;
}

double* Upper_triangular_invert(double** U,double* X,int size){  //solves UX=Y
	int i,j;
	double* Y = clone_vector(X,size);
	for (i=size-1;i>=0;i--){
		for (j=i+1;j<size;j++) Y[i] -= U[i][j]*Y[j];  
		if (U[i][i]==0){
			printf("U inversion not possible: zero diagonal at index %d\n",i);
			exit(0);
		}
		Y[i] /= U[i][i];
	}
	return Y;
}

double* Upper_triangular_transpose_invert(double** U,double* X,int size){  //solves U^TX=Y
	int i,j;
	double* Y = clone_vector(X,size);
	for (i=0;i<size;i++){
		//Y[i] = X[i];
		for (j=0;j<i;j++) Y[i] -= U[j][i]*Y[j];  
		if (U[i][i]==0){
			printf("U inversion not possible: zero diagonal at index %d\n",i);
			exit(0);
		}
		Y[i] /= U[i][i];
	}
	return Y;
}

void LU_factorzation(double** A,double** L,double** U,int size){
	int i,j,sub_size;
	double diag;
	double* a = NULL;
	double* u = NULL;
	double* l = NULL;
	for (i=0;i<size;i++){
		L[i] = (double*)malloc(size*sizeof(double));
		U[i] = (double*)malloc(size*sizeof(double));
		for (j=0;j<size;j++) {
			L[i][j] = 0;
			U[i][j] = 0;
		}
	}
	L[0][0] = 1.;
	U[0][0] = A[0][0];
	for (sub_size=1;sub_size<size;sub_size++){
		a = (double*)realloc(a,sub_size*sizeof(double));
		for (i=0;i<sub_size;i++) a[i] = A[i][sub_size];
		u = Lower_triangular_invert(L,a,sub_size);
		l = Upper_triangular_transpose_invert(U,A[sub_size],sub_size);
		for (i=0;i<sub_size;i++){
			U[i][sub_size] = u[i];
			L[sub_size][i] = l[i];
		}
		diag = A[sub_size][sub_size];
		for (i=0;i<sub_size;i++) diag -= u[i]*l[i];
		U[sub_size][sub_size] = diag;
		L[sub_size][sub_size] = 1.;
		free(u);
		free(l);
	}
	free(a);
}

void test_LU(double** A,double** L,double** U,int size){
	int i,j,k;
	double sum,sum_A;
	double** B = (double**)malloc(size*sizeof(double*));
	for (i=0;i<size;i++){
		B[i] = (double*)malloc(size*sizeof(double));
		for (j=0;j<size;j++){
			B[i][j] = 0;
			for (k=0;k<size;k++) B[i][j] += L[i][k]*U[k][j];
		}
	}
	sum = 0;
	sum_A = 0;
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			sum += fabs(A[i][j]-B[i][j]);
			sum_A += fabs(A[i][j]);
		}
	}
	printf("LU-deviation: %f\n",sum/sum_A);
}

int cholesky_factorization(double** A,double** C, int size){
	int i,j,k;
	for (i=0;i<size;i++){
		C[i] = (double*)malloc(size*sizeof(double));
		for (j=0;j<size;j++) C[i][j] = 0;
	}
	for (i=0;i<size;i++){
		for (j=0;j<i;j++){
			C[i][j] = A[i][j];
			for (k=0;k<j;k++) C[i][j] -= C[i][k]*C[j][k];
			C[i][j] /= C[j][j];
		}
		C[i][i] = A[i][i];
		for (k=0;k<i;k++) C[i][i] -= C[i][k]*C[i][k];
		if (C[i][i]<0) return 0;
		C[i][i] = sqrt(C[i][i]);
	}
	return 1;
}

int sparse_element_number(sparse_matrix* A,int i){
	int j;
	int counter = 0;
	for (j=0;j<A->Len[i];j++) if (A->Values[i][j]!=0) counter++;
	return counter;
}

double** sub_convert_sparse_to_array(sparse_matrix* A,int* Map,int* Inv_Map,int* new_size){
	int i,j,k,I,J; 
	int n = A->size;
	double diag;
	double** M_A = NULL;
	I = 0;
	for (i=0;i<n;i++){
		diag = get_diag(A,i);
		if (sparse_element_number(A,i)>1){
			M_A = (double**)realloc(M_A,(I+1)*sizeof(double*));
			M_A[I] = zero_vector(n);
			for (j=0;j<A->Len[i];j++){
				k = A->Indices[i][j];
				M_A[I][k] = A->Values[i][j];
			}
			Map[i] = I;
			Inv_Map[I] = i;
			I++;
		}
		else Map[i] = -1;
	}
	*new_size = I;
	double** M_B = (double**)malloc((*new_size)*sizeof(double*));
	for (I=0;I<(*new_size);I++){
		M_B[I] = zero_vector(*new_size);
		for (J=0;J<(*new_size);J++) M_B[I][J] = M_A[I][Inv_Map[J]];
	}
	for (I=0;I<(*new_size);I++) free(M_A[I]);
	free(M_A);
	return M_B;
}

void sub_convert_array_to_sparse(sparse_matrix* A,int* Map,int* Inv_Map,double** S_L,
 double** S_U,sparse_matrix* L,sparse_matrix* U,int sub_size){		// triviale Zeilen müssen diagonal sein !
	int i,j;
	for (i=0;i<sub_size;i++){
		for (j=0;j<sub_size;j++){
			if (S_L[i][j]!=0) insert_sparse(L,S_L[i][j],Inv_Map[i],Inv_Map[j]);
			if (S_U[i][j]!=0) insert_sparse(U,S_U[i][j],Inv_Map[i],Inv_Map[j]);
		}
	}
	for (i=0;i<A->size;i++) if (Map[i]<0){
		insert_sparse(L,1.,i,i);
		insert_sparse(U,get_diag(A,i),i,i);
	}
 }

void sparse_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	int i;
	int n = A->size;
	int* N = (int*)malloc(sizeof(int));
	int* Map = (int*)malloc(n*sizeof(int));
	int* Inv_Map = (int*)malloc(n*sizeof(int));
	double** Sub = sub_convert_sparse_to_array(A,Map,Inv_Map,N);
	double** S_U = (double**)malloc((*N)*sizeof(double*));
	double** S_L= (double**)malloc((*N)*sizeof(double*));
	LU_factorzation(Sub,S_L,S_U,*N);
	sub_convert_array_to_sparse(A,Map,Inv_Map,S_L,S_U,L,U,*N);
	for (i=0;i<(*N);i++){
		free(Sub[i]);
		free(S_U[i]);
		free(S_L[i]);
	}
	free(Sub);
	free(S_U);
	free(S_L);
	free(Map);
	free(Inv_Map);
	free(N);
}

double** convert_sparse_to_array(sparse_matrix* A,int col_size){
	int i,j,k;
	int n = A->size;
	double** M_A = (double**)malloc(n*sizeof(double*));
	for (i=0;i<n;i++){
		M_A[i] = zero_vector(col_size);
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			if (k<col_size) M_A[i][k] = A->Values[i][j];
			else{
				printf("sparse->array: wrong size -> abort\n");
				exit(0);
			}
		}
	}
	return M_A;
}

void convert_array_to_sparse(double** M_A,sparse_matrix* A,int col_size,int row_size){
	int i,j;  // sparse zero reingeben 
	for (i=0;i<col_size;i++){
		for (j=0;j<row_size;j++) if (M_A[i][j]!=0){
			insert_sparse(A,M_A[i][j],i,j);
		}
	}
}

void convert_transposed_array_to_sparse(double** M_A,sparse_matrix* A,int col_size,int row_size){
	int i,j;  // sparse zero reingeben 
	for (i=0;i<col_size;i++){
		for (j=0;j<row_size;j++) if (M_A[i][j]!=0){
			insert_sparse(A,M_A[i][j],j,i);
		}
	}
}

int square_matrix_clean(sparse_matrix* A){
	int i,j,ind;
	double val;
	int n = A->size;
	for (i=0;i<n;i++){
		j = 0;
		while (j<A->Len[i]){
			val = A->Values[i][j];
			ind = A->Indices[i][j];
			if (ind>=n){
				if (val==0) remove_element_at(A,i,j);
				else return 0;
			}
			else{
				if (val==0 && ind!=i) remove_element_at(A,i,j);
				else j++;
			}
		}
	}
	return 1;
}

void complete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	int i;   // L und U = sparse zero reingeben
	int n = A->size;
	double** M_A = convert_sparse_to_array(A,n);
	double** L_A = (double**)malloc(n*sizeof(double*));
	double** U_A = (double**)malloc(n*sizeof(double*));
	LU_factorzation(M_A,L_A,U_A,n);
	convert_array_to_sparse(L_A,L,n,n);
	convert_array_to_sparse(U_A,U,n,n);
	for (i=0;i<n;i++) {
		free(M_A[i]);
		free(L_A[i]);
		free(U_A[i]);
	}
	free(M_A);
	free(L_A);
	free(U_A);
}

int cholesky(sparse_matrix* A,sparse_matrix* C,sparse_matrix* CT){
	int i;   // C = sparse zero reingeben, CT sparse zero oder NULL
	int n = A->size;
	double** M_A = convert_sparse_to_array(A,n);
	double** C_A = (double**)malloc(n*sizeof(double*));
	int res = cholesky_factorization(M_A,C_A,n);
	convert_array_to_sparse(C_A,C,n,n);
	if (CT!=NULL) convert_transposed_array_to_sparse(C_A,CT,n,n);
	for (i=0;i<n;i++) {
		free(M_A[i]);
		free(C_A[i]);
	}
	free(M_A);
	free(C_A);
	return res;
}

double* zero_vector(int n){
	int i;
	double* Res = (double*)malloc(n*sizeof(double));
	//#ifdef _OPENMP
	//#pragma omp parallel for
	//#endif
	for (i=0;i<n;i++){Res[i] = 0;}
	return Res;
}

int* zero_int_list(int n){
	int i;
	int* Res = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++){Res[i] = 0;}
	return Res;
}

double** zero_matrix(int n,int m){
	int i;
	double** Res = (double**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++) Res[i] = zero_vector(m);
	return Res;
}

void convert_sparse_to_UMFPACK(sparse_matrix* A,int col_num,int** Col_start,int** Indices,double** Values,int* Size){
	int i,j,k,s,N;
	double aji;
	sparse_matrix* AT = get_transpose(A,col_num);
	//int n = A->size;
	*Col_start = (int*)malloc((col_num+1)*sizeof(int));
	N = 0;
	for (i=0;i<col_num;i++) N += AT->Len[i];
	*Indices = (int*)malloc(N*sizeof(int));
	*Values = (double*)malloc(N*sizeof(double));
	
	k = 0;
	for (i=0;i<col_num;i++){
		(*Col_start)[i] = k;
		for (j=0;j<AT->Len[i];j++){
			aji = AT->Values[i][j];
			if (aji!=0){
				(*Indices)[k] = AT->Indices[i][j];
				(*Values)[k] = aji;
				k++;
			}
		}
	}
	(*Col_start)[col_num] = k;
	*Size = k;
	if (k<N){
		*Indices = (int*)realloc(*Indices,k*sizeof(int));
		*Values = (double*)realloc(*Values,k*sizeof(double));
	}
	free_sparse(AT);
}

void convert_UMFPACK_to_sparse(sparse_matrix** A,int col_num,int row_num,int* Col_start,int* Indices,double* Values){
	int i,j;
	*A = sparse_zero(row_num);
	for (i=0;i<col_num;i++){
		for (j=Col_start[i];j<Col_start[i+1];j++){
			insert_sparse(*A,Values[j],Indices[j],i);
		}
	}
}

int* generate_list(int* var_indices,int var_len,int grid_size,int whole_size){
	int i,j,k,start,dest;
	int len = whole_size-var_len*grid_size;
	int* Res = (int*)malloc(len*sizeof(int));
	k = 0;
	start = 0;
	for (i=0;i<=var_len;i++){
		if (i==var_len){dest = whole_size;}else{dest = var_indices[i]*grid_size;}
		for (j=start;j<dest;j++){
			if (k>=len){printf("Fehler bei generate_list index %d\n",j);}
			Res[k] = j;
			k++;
		}	
		start = dest+grid_size;	
	}
	return Res;
}	

sparse_matrix* restrict_matrix_rows(sparse_matrix* A,int* List,int list_size,int* index_map){
	int i,j,k,ind;	  // Vorsicht: Die Elemente in List werden gelöscht !
	sparse_matrix* Res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	Res->size = A->size-list_size;
	Res->Len = (int*)malloc(Res->size*sizeof(int));
	Res->Values = (double**)malloc(Res->size*sizeof(double*));
	Res->Indices = (int**)malloc(Res->size*sizeof(int*));
	for (i=0;i<A->size;i++){index_map[i] = -1;}
	
	int start = 0;
	int counter = 0;
	for (i=0;i<=list_size;i++){
		if (i<list_size){ind = List[i];}else{ind = A->size;}
		for (j=start;j<ind;j++){			
			index_map[j] = counter;
			Res->Len[counter] = A->Len[j];
			Res->Values[counter] = (double*)malloc(A->Len[j]*sizeof(double));
			Res->Indices[counter] = (int*)malloc(A->Len[j]*sizeof(int));
			for (k=0;k<A->Len[j];k++){
				Res->Values[counter][k] = A->Values[j][k];
				Res->Indices[counter][k] = A->Indices[j][k];
			}
			counter++;
		}
		start = ind+1;
	}
	return Res;	
}

double* sparse_row_sum(sparse_matrix* A){
	int i,j;	
	double* Res = zero_vector(A->size);
	for (i=0;i<A->size;i++){		
		for (j=0;j<A->Len[i];j++) Res[i] += A->Values[i][j];
	}
	return Res;
}

double* restrict_matrix_cols(sparse_matrix* A,double* Sol,int* index_map){
	int i,j,ind;	
	int n = A->size;
	double* Res = zero_vector(A->size);
	for (i=0;i<n;i++){
		j = 0;		
		do{			
			ind = A->Indices[i][j];
			if (index_map[ind]>=0){
				A->Indices[i][j] = index_map[ind];
				j++;
			}
			else{
				Res[i] -= A->Values[i][j]*Sol[A->Indices[i][j]];
				remove_element_at(A,i,j);
			}									
		}while(j<A->Len[i]);			
	}
	return Res;
}

sparse_matrix* restrict_matrix(sparse_matrix* A,double* b,double** b_sub,int** index_map,
double* Solution,double* Prev_solution,double tol){
	int i;
	int list_size = 0;
	int* List = (int*)malloc(0);
	for (i=0;i<A->size;i++){
		if (fabs(Solution[i]-Prev_solution[i])<tol){
			list_size++;
			List = (int*)realloc(List,list_size*sizeof(int));
			List[list_size-1] = i;
		}
	}
		
	index_map[0] = (int*)malloc(A->size*sizeof(int));
	for (i=0;i<A->size;i++){index_map[0][i] = -1;}
	sparse_matrix* A_sub = restrict_matrix_rows(A,List,list_size,index_map[0]);
	
	*b_sub = restrict_matrix_cols(A_sub,Solution,index_map[0]);
	for (i=0;i<A->size;i++){
		if (index_map[0][i]>=0){b_sub[0][index_map[0][i]] +=b[i];}		
	}		
	
	free(List);	
	return A_sub;	
}

double* restrict_system(sparse_matrix* A,double* F,double* Sol,int start,int end){
	int i,j,k;
	int n = A->size;
	double* New_F = zero_vector(end-start);
	for (i=start;i<end;i++){
		New_F[i-start] = F[i];
		j = 0;
		while (j<A->Len[i]){
			k = A->Indices[i][j];
			if (k<start || k>=end) {
				if (Sol!=NULL) New_F[i-start] -= A->Values[i][j]*Sol[k];
				remove_element_at(A,i,j);
			}
			else j++;
		}
	}
	for (i=start;i<end;i++){
		if (i-start<start){
			free(A->Indices[i-start]);
			free(A->Values[i-start]);
		}
		A->Indices[i-start] = A->Indices[i];
		A->Values[i-start] = A->Values[i];
		A->Len[i-start] = A->Len[i];
	}
	for (i=end-start;i<start;i++){
		free(A->Indices[i]);
		free(A->Values[i]);
	}
	for (i=end;i<n;i++){
		free(A->Indices[i]);
		free(A->Values[i]);
	}
	A->size = end-start;
	A->Len = (int*)realloc(A->Len,(end-start)*sizeof(int));
	A->Indices = (int**)realloc(A->Indices,(end-start)*sizeof(int*));
	A->Values = (double**)realloc(A->Values,(end-start)*sizeof(double*));
	for (i=0;i<end-start;i++){
		for (j=0;j<A->Len[i];j++) A->Indices[i][j] -= start;
	}
	return New_F;
}

double* restrict_vector(double* Vector,int* index_map,int n_sol,int n_res){
	int i;
	double* Res = (double*)malloc(n_res*sizeof(double));
	for (i=0;i<n_sol;i++){
		if (index_map[i]>=0){
			if (index_map[i]>=n_res){printf("fehlerhafte Index_map bei %d\n",i);}
			Res[index_map[i]] = Vector[i];			
		}
	}
	return Res;
}

double* restrict_vector_by(int i_min,int i_max,double* Vector){
	int i;
	double* Res = (double*)malloc((i_max-i_min)*sizeof(double));
	for (i=i_min;i<i_max;i++){
		Res[i-i_min] = Vector[i];
	}
	return Res;	
}

void insert_vector(double* Vector,double* vector,int start,int size){
	int i;
	for (i=start;i<start+size;i++){
		Vector[i] = vector[i-start];
	}	
}

void map_back(double* Solution,double* Restricted,int n_sol,int* index_map){
	int i;
	for (i=0;i<n_sol;i++){
		if (index_map[i]>=0){
			Solution[i] = Restricted[index_map[i]];
		}		
	}	
}

void set_zero(double* Vector,int start,int end){
	int i;
	for (i=start;i<end;i++){Vector[i] = 0;}
}

double* get_resized_vector(double* Vector,int old_size,int new_size){
	int i;
	double* Res = (double*)malloc(new_size*sizeof(double));
	for (i=0;i<old_size;i++){Res[i] = Vector[i];}
	for (i=old_size;i<new_size;i++){Res[i] = 0;}
	return Res;
}

double* join_vectors(double* V1,int n1,double* V2,int n2){
	int i;
	double* Res = (double*)malloc((n1+n2)*sizeof(double));
	for (i=0;i<n1;i++){Res[i] = V1[i];}
	for (i=n1;i<n1+n2;i++){Res[i] = V2[i-n1];}
	return Res;
}
	
void append_vector(double** V1,int n1,double* V2,int n2){
	int i;
	V1[0] = (double*)realloc(V1[0],(n1+n2)*sizeof(double));
	for (i=n1;i<n1+n2;i++){V1[0][i] = V2[i-n1];}	
}

double get_diag(sparse_matrix* A,int i){
	int flag = -1;	
	int pos = get_position(A->Indices[i],A->Len[i],i,&flag);
	if (flag<0){return 0;}
	else{return A->Values[i][pos];}
}

double matrix_norm(sparse_matrix* A){
	int i,j;
	double sum = 0;
	for (i=0;i<A->size;i++){		
		for (j=0;j<A->Len[i];j++){
			sum += fabs(A->Values[i][j]);
		}
	}
	return (double)sum/A->size;
}

double get_largest_matrix_element(sparse_matrix* A){
	int i,j;
	int I = -1;
	int J = -1;
	double max = 0;
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++) if (fabs(A->Values[i][j])>max){
			max = fabs(A->Values[i][j]);
			I = i;
			J = A->Indices[i][j];
		}
	}
	printf("largest found at: (%d|%d)\n",I,J);
	return max;
}

double estimate_operator_norm(sparse_matrix* A){ // gives upper bound to <x,Ax> with <x,x>=1
	int i,j;
	double sum;
	double max = 0;
	for (i=0;i<A->size;i++){		
		sum = 0;
		for (j=0;j<A->Len[i];j++){
			sum += fabs(A->Values[i][j]);
		}
		if (sum>max) max = sum;
	}
	return max*sqrt((double)A->size);
}

double get_worst_diagonal(sparse_matrix* A){
	int i,j;
	int n = A->size;
	int i_max = -1;
	double sum,diag;
	double max = 0;
	for (i=0;i<n;i++){
		sum = 0;
		diag = 0;
		for (j=0;j<A->Len[i];j++) if (A->Indices[i][j]!=i) sum += fabs(A->Values[i][j]);
			else diag = fabs(A->Values[i][j]);
		if (diag!=0){
			/*if (sum/diag>max){
				max = sum/diag;
				i_max = i;
			}*/
			for (j=0;j<A->Len[i];j++) if (A->Indices[i][j]!=i){
				sum = A->Values[i][j];
				if (sum/diag>max){
					max = sum/diag;
					i_max = i;
				}
			}
		}
		else{
			max = sum/diag;
			i_max = i;
			break;
		}
	}
	printf("worst index: %d\n",i_max);
	return max;
}

int test_pos(int* list,int n,int ind){
	int i;
	for (i=0;i<n;i++){
		if (list[i]==ind){
			return i;
		}
	}
	return -1;
}

/*sparse_matrix* get_test_matrix(){
	sparse_matrix* Res = sparse_identity(2);
	insert_sparse(Res,0.5,0,1);
	insert_sparse(Res,0.5,1,0);
	return Res;
}*/

sparse_matrix* get_givens_rotation(int i, int j, double alpha, int n){
	sparse_matrix* Res = sparse_identity(n);
	insert_sparse(Res,cos(alpha)-1.,i,i);
	insert_sparse(Res,cos(alpha)-1.,j,j);
	insert_sparse(Res,-sin(alpha),i,j);
	insert_sparse(Res,sin(alpha),j,i);
	return Res;
}

void perform_givens_rotation(sparse_matrix* A,int i,int j,double c,double s){	// multiplies with {{c,s},{-s,c}}, s^2+c^2=1
	int k,l,m;
	int max_len = A->Len[i]+A->Len[j];
	double* Val_i = zero_vector(max_len);
	double* Val_j = zero_vector(max_len);
	int* Ind_i = zero_int_list(max_len);
	int* Ind_j = zero_int_list(max_len);
	m = 0;
	l = 0;
	for(k=0;k<A->Len[i];k++){
		while (l<A->Len[j] && A->Indices[j][l]<=A->Indices[i][k]){			
			Val_i[m] += s*A->Values[j][l];
			Val_j[m] += c*A->Values[j][l];
			Ind_i[m] = A->Indices[j][l];
			Ind_j[m] = Ind_i[m];
			if (A->Indices[j][l]!=A->Indices[i][k]) m++;
			l++;
		}		
		Val_i[m] += c*A->Values[i][k];
		Val_j[m] += -s*A->Values[i][k];
		Ind_i[m] = A->Indices[i][k];
		Ind_j[m] = Ind_i[m];
		m++;
	}
	
	for (k=l;k<A->Len[j];k++){
		Val_i[m] += s*A->Values[j][k];
		Val_j[m] += c*A->Values[j][k];
		Ind_i[m] = A->Indices[j][k];
		Ind_j[m] = Ind_i[m];
		m++;
	}
	if (m>max_len){
		printf("error in Givens rotation, coordinate (%d,%d) -> abort\n",i,j);
		exit(0);
	}
	Val_i = (double*)realloc(Val_i,m*sizeof(double));
	Val_j = (double*)realloc(Val_j,m*sizeof(double));
	Ind_i = (int*)realloc(Ind_i,m*sizeof(int));
	Ind_j = (int*)realloc(Ind_j,m*sizeof(int));		
	
	A->Len[i] = m;
	A->Len[j] = m;
	free(A->Indices[i]);
	free(A->Indices[j]);
	A->Indices[i] = Ind_i;
	A->Indices[j] = Ind_j;
	free(A->Values[i]);
	free(A->Values[j]);
	A->Values[i] = Val_i;
	A->Values[j] = Val_j;
}

void Hessenberg_QR(sparse_matrix* H,sparse_matrix** R,sparse_matrix** Q){		// computes Q.R = H, 
	int i,j;																	// H upper Hessenberg matrix
	double a,c,r;
	int n = H->size;
	sparse_matrix* QT = sparse_identity(n);
	*R = clone(H);
	for (i=0;i<n-1;i++){	
		a = get_diag(*R,i);
		c = get_matrix_element(*R,i+1,i);
		r = sqrt(a*a+c*c);
		if (r!=0){
			a /= r;
			c /= r;
			perform_givens_rotation(*R,i,i+1,a,c);
			perform_givens_rotation(QT,i,i+1,a,c);
		}	
	}
	*Q = get_transpose(QT,n);
	free_sparse(QT);
}

void QR_decomposition(sparse_matrix* A,sparse_matrix** R,sparse_matrix** Q){		// computes Q.R = H, 
	int i,j;																		// A square matrix
	double a,c,r;
	int n = A->size;
	sparse_matrix* QT = sparse_identity(n);
	*R = clone(A);
	for (i=0;i<n-1;i++){	
		for (j=i+1;j<n;j++){		
			a = get_diag(*R,i);
			c = get_matrix_element(*R,j,i);		
			r = sqrt(a*a+c*c);			
			if (c!=0 && r!=0){
				a /= r;
				c /= r;
				perform_givens_rotation(*R,i,j,a,c);
				perform_givens_rotation(QT,i,j,a,c);
			}		
		}
	}
	*Q = get_transpose(QT,n);
	free_sparse(QT);
}

void Ritz_values(sparse_matrix* H,double** V,double* Eig,int iter){		
	const double tol = 1e-8;
	int i,j,k;
	int n = H->size;
	sparse_matrix* P = NULL;
	sparse_matrix* Q = NULL;
	sparse_matrix* R = NULL;
	sparse_matrix* A = clone(H);
	sparse_matrix* S = sparse_identity(n);
	
	// QR iteration to obtain approximately triangular matrix
	for (i=0;i<iter;i++){
		Hessenberg_QR(A,&R,&Q);
		free_sparse(A);
		A = sparse_product(R,Q);
		sparse_approximate(A,tol);
		
		P = sparse_product(S,Q);
		free_sparse(S);
		S = P;
		
		free_sparse(R);
		free_sparse(Q);		
	}
																			// H = S.A.ST oder A = ST.H.S	
	double** B = convert_sparse_to_array(A,n);
	
	// eigenvalues
	for (i=0;i<n;i++) Eig[i] = B[i][i];
	
	// eigenvectors
	V[0] = zero_vector(n);
	V[0][0] = 1.;
	for (i=1;i<n;i++){
		V[i] = zero_vector(n);
		// upper triangular invert (if triangular not well approximated maybe use Gauss-Seidel ?)
		for (j=i-1;j>=0;j--){
			V[i][j] = B[j][i];
			for (k=j+1;k<i;k++) V[i][j] -= B[j][k];
			if (B[j][j]!=B[i][i]) V[i][j] /= B[j][j]-B[i][i];
			else{
				printf("error in triangular inversion in QR_eigenvalues -> abort\n");
				exit(0);			
			}			
		}
		V[i][i] = 1.;		
		vector_normalize(V[i],n);		
	}
	
	for (i=0;i<n;i++){
		sparse_multiplication(S,V[i]);
		free(B[i]);
	}
	
	for (i=0;i<n;i++){
		double* X = sparse_mult(H,V[i]);
		vector_add(X,V[i],-Eig[i],n);
		printf("deviation eig %d: %e\n",i,euklid_norm(X,n));
		free(X);
	}
	
	
	free(B);
	free_sparse(A);
	free_sparse(S);
}

void Harmonic_Ritz_values(sparse_matrix* A,double** V,double* V0,double* Eig,int s){		
	const double tol = 1e-8;
	const int iter = 100;
	int i,j,k;
	double rii,rji,r;
	int n = A->size;
	double** Z = (double**)malloc(s*sizeof(double*));
	double** W = (double**)malloc(s*sizeof(double*));
	sparse_matrix* R = sparse_zero(s);
	for (i=0;i<s;i++){
		if (i>0) W[i] = sparse_mult(A,W[i-1]); else W[i] = sparse_mult(A,V0);
		for (j=0;j<i;j++){
			rji = scalar(W[i],W[j],n);
			vector_add(W[i],W[j],-rji,n);
			insert_sparse(R,rji,j,i);
		}
		rii = sqrt(scalar(W[i],W[i],n));
		scalar_mult(1./rii,W[i],n);
		insert_sparse(R,rii,i,i);
		if (i>0) Z[i] = clone_vector(W[i-1],n); else Z[i] = clone_vector(V0,n);
	}
	
	sparse_matrix* WTZ = sparse_zero(s);
	for (i=0;i<s;i++){
		insert_sparse(WTZ,scalar(V0,W[i],n),i,0);
		if (i<s-1) insert_sparse(WTZ,1.,i,i+1);
	}
	
	sparse_matrix* Inv_R = get_inv_U_triang(R);
	print_sparse(Inv_R);	
	sparse_matrix* B = sparse_product(WTZ,Inv_R);
	sparse_matrix* H = clone(B);
	
	sparse_matrix* B_R = NULL;
	sparse_matrix* Q = NULL;
	sparse_matrix* P = NULL;
	sparse_matrix* S = sparse_identity(s);	
	
	for (i=0;i<iter;i++){													// implement case for complex eigenvalues
		QR_decomposition(B,&B_R,&Q);
		free_sparse(B);
		B = sparse_product(B_R,Q);
		P = sparse_product(S,Q);
		free_sparse(S);
		S = P;
		
		free_sparse(B_R);
		free_sparse(Q);		
	}
	sparse_approximate(B,tol);
	print_sparse(B);
	
	double** U = (double**)malloc(s*sizeof(double*));
	for (i=0;i<s;i++){
		Eig[i] = 1./get_diag(B,i);
		U[i] = zero_vector(s);
		U[i][i] = 1.;
		sparse_multiplication(S,U[i]);			
	}
	
	for (i=0;i<s;i++){
		V[i] = zero_vector(n);
		for (j=0;j<s;j++) vector_add(V[i],W[j],U[i][j],n);
		vector_normalize(V[i],n);
	}
	
	double d;
	sparse_matrix* ID = sparse_identity(s);
	for (i=0;i<s;i++){		
		if (i==0) d = -1./Eig[0]; else d = 1./Eig[i-1]-1./Eig[i];
		sparse_add(H,ID,d);
		double* X = sparse_mult(H,U[i]);
		printf("deviation eig %d: %e\n",i,euklid_norm(X,s)/matrix_norm(H));
		free(X);
	}
	
	free_sparse(H);
	free_sparse(ID);
	free_sparse(R);
	free_sparse(Inv_R);
	free_sparse(WTZ);
	free_sparse(S);
	
	for (i=0;i<s;i++){
		free(W[i]);
		free(Z[i]);
		free(U[i]);
	}
	free(W);
	free(Z);
	free(U);
}

void Arnoldi(sparse_matrix* A,sparse_matrix** H,double** U,double* V0,int* s){				
	const double tol = 1e-10;
	int i,j;
	double h_sub;
	int n = A->size;
	sparse_matrix* HT = sparse_zero(*s);
	double* D = zero_vector(n);
	copy_vector_to(V0,D,n);
	h_sub = euklid_norm(D,n);
	
	for (i=0;i<*s;i++){
		if (h_sub<tol){
			*s = i;
			printf("Arnoldi: subdiagonal smaller than tolerance after %d steps-> terminate\n",*s);
			break;
		}
		U[i] = clone_vector(D,n);
		scalar_mult(1./h_sub,U[i],n);
		free(D);
		D = sparse_mult(A,U[i]);
		for (j=0;j<=i;j++) insert_sparse(HT,scalar(D,U[j],n),i,j);
		for (j=i;j>=0;j--) vector_add(D,U[j],-HT->Values[i][j],n);
		h_sub = euklid_norm(D,n);
		if (i+1<*s) insert_sparse(HT,h_sub,i,i+1);
	}
	*H = get_transpose(HT,*s);
	free_sparse(HT);
	free(D);
}

void reortho_Arnoldi(sparse_matrix* A,sparse_matrix** H,double** U,double* V0,int* s,int iter){				
	const double tol = 1e-10;
	int i,j,k;
	double h_sub,hij;
	int n = A->size;
	sparse_matrix* HT = sparse_zero(*s);
	double* D = zero_vector(n);
	copy_vector_to(V0,D,n);
	h_sub = euklid_norm(D,n);
	
	for (i=0;i<*s;i++){			
		if (h_sub<tol){
			*s = i;
			printf("Arnoldi: subdiagonal smaller than tolerance after %d steps-> terminate\n",*s);
			break;
		}
		U[i] = clone_vector(D,n);
		scalar_mult(1./h_sub,U[i],n);
		free(D);
		D = sparse_mult(A,U[i]);		
		for (k=0;k<iter;k++){			
			for (j=0;j<=i;j++){
				hij = scalar(D,U[j],n);
				vector_add(D,U[j],-hij,n);
				if (k==0) insert_sparse(HT,hij,i,j);	
			}							
		}
		h_sub = euklid_norm(D,n);
		if (i+1<*s) insert_sparse(HT,h_sub,i,i+1);
	}
	*H = get_transpose(HT,*s);
	free_sparse(HT);
	free(D);
}

sparse_matrix* get_test_matrix(int n,int random_rot){
	int i,j,k;
	double a;
	double* Eig = zero_vector(n);
	for (i=0;i<n;i++) Eig[i] = (double)(i+1)/n;
	//for (i=0;i<10;i++) Eig[n/2+i] = (double)(i+1)/n-0.1;
	Eig[n/2] = -Eig[n/2];
	print_vector(Eig,n);
	//srand(time(NULL));
	srand(0);
	sparse_matrix* Res = sparse_diagonal(Eig,n);
	for (k=0;k<random_rot;k++){
		i = rand() % n;
		j = rand() % n;
		if (i!=j){
			a = (double)2.*M_PI*rand()/RAND_MAX;
			sparse_matrix* U = get_givens_rotation(i,j,a,n);
			sparse_matrix* UT = get_transpose(U,n);
			sparse_right_mult(Res,UT);
			sparse_left_mult(U,Res);
			free_sparse(U);
			free_sparse(UT);
		}
	}
	free(Eig);
	return Res;
}

void sparse_substitute(sparse_matrix* A,int p,int q,int J,double* F){
	int j;
	double ap = 0;
	double aq = 0;
	for (j=0;j<A->Len[p];j++) if (A->Indices[p][j]==J){
		ap = A->Values[p][j];
		break;
	}
	for (j=0;j<A->Len[q];j++) if (A->Indices[q][j]==J){
		aq = A->Values[q][j];
		break;
	}
	if (ap!=0 && aq!=0){
		for (j=0;j<A->Len[p];j++) insert_sparse(A,-(A->Values[p][j])*aq/ap,q,A->Indices[p][j]);
		if (F!=NULL) F[q] -= F[p]*aq/ap;
		remove_element(A,q,J);
	}
}

void make_nonzero_diag(sparse_matrix* A,double* F){
	int i,j,k,imax,jmax;
	double a,b,d,max;
	int n = A->size;
	for (i=0;i<n;i++){
		d = get_diag(A,i);
		if (d==0){
			imax = -1;
			jmax = -1;
			max = 0;
			for (j=0;j<A->Len[i];j++){
				a = fabs(A->Values[i][j]);
				if (a!=0){
					for (k=0;k<n;k++) if (k!=i){
						b = fabs(get_matrix_element(A,k,i));
						if (b!=0) if (a/b>max){
							max = a/b;
							imax = k;
							jmax = A->Indices[i][j];
						}
					}
				}
			}
			if (imax>0) sparse_substitute(A,imax,i,jmax,F);
		}
	}
}

void Gauss_elemination(sparse_matrix* Matrix,double* Vector,double* Solution,double tol){		
	int i,j,k,pos;
	int flag = 0;
	sparse_matrix* A = clone(Matrix);
	int n = A->size;
	double* b = clone_vector(Vector,n);
	double diag,aji,fac;
	for (i=0;i<n;i++){
		diag = get_diag(A,i);
		for (j=0;j<n;j++){
			pos = get_position(A->Indices[j],A->Len[j],i,&flag);			
			aji = A->Values[j][pos];
			if (j!=i && flag>=0){
				fac = -aji/diag;
				for (k=0;k<A->Len[i];k++){
					insert_sparse(A,A->Values[i][k]*fac,j,A->Indices[i][k]);
				}
				b[j] += b[i]*fac;	
				pos = get_position(A->Indices[j],A->Len[j],i,&flag);
				if (fabs(A->Values[j][pos])<tol){remove_element_at(A,j,pos);}	
				else{printf("Fehler Zeile %d\n",j);}			
			}				
		}		
	}
	for (i=0;i<n;i++){Solution[i] = b[i]/get_diag(A,i);}
	free(b);
	free_sparse(A);		
}

void change_row(sparse_matrix* A,double* F,int i1,int i2){
	double* Val = A->Values[i1];
	int* Ind = A->Indices[i1];
	int len = A->Len[i1];
	A->Values[i1] = A->Values[i2];
	A->Values[i2] = Val;
	A->Indices[i1] = A->Indices[i2];
	A->Indices[i2] = Ind;
	A->Len[i1] = A->Len[i2];
	A->Len[i2] = len;
	double f = F[i1];
	F[i1] = F[i2];
	F[i2] = f;
}

void change_var_rows(sparse_matrix* A,double* F,int var1,int var2,int block_size){
	int i;
	//double s;
	for (i=0;i<block_size;i++){
		//s = Sol[block_size*var1+i];
		//Sol[block_size*var1+i] = Sol[block_size*var2+i];
		//Sol[block_size*var2+i] = s;
		change_row(A,F,block_size*var1+i,block_size*var2+i);
	}
}

void change_to_spatial_ordering(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int block_size){
	int i,j,k,i_mesh,i_var,j_mesh,j_var;
	int n = A->size;
	int varnum = n / block_size;
	for (i=0;i<n;i++){
		i_mesh = i % block_size;
		i_var = i / block_size;
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			j_mesh = k % block_size;
			j_var = k / block_size;
			insert_sparse(B,A->Values[i][j],varnum*i_mesh+i_var,varnum*j_mesh+j_var);
		}
		G[varnum*i_mesh+i_var] = F[i];
	}
}

void map_indices(sparse_matrix* A,sparse_matrix* B,double* F,double* G,int* Map){
	int i,j,k;
	int n = A->size;
	for (i=0;i<n;i++){
		for (k=0;k<A->Len[i];k++){
			j = A->Indices[i][k];
			insert_sparse(B,A->Values[i][k],Map[i],Map[j]);
		}
		if (F!=NULL && G!=NULL) G[Map[i]] = F[i];
	} 
}

double* generate_vector(int size,double val){
	int i;
	double* Res = (double*)malloc(size*sizeof(double));
	for (i=0;i<size;i++){Res[i] = val;}
	return Res;
}

/*void resize_matrix(sparse_matrix* A,int new_size){
	int i,j;
	int old_size = A->size;
	int** Index_buffer = (int**)malloc(old_size*sizeof(int*));
	double** Val_buffer = (double**)malloc(old_size*sizeof(double*));
	for (i=0;i<old_size;i++){
		Index_buffer[i] = (int*)malloc(A->Len[i]*sizeof(int));
		Val_buffer[i] = (double**)malloc(A->Len[i]*sizeof(double));
		for (j=0;j<A->Len[i];j++){
			Index_buffer[i][j] = A->Indices[i][j];
			Val_buffer[i][j] = A->Values[i][j];
		}
	}
	
	
	
	
	A->size = new_size;
	printf("val %f index %d\n",A->Values[0][0],0); 
	double* T = A->Values[0];
	A->Len = (int*)realloc(A->Len,new_size);
	A->Indices = (int**)realloc(A->Indices,new_size*sizeof(int*));
	A->Values = (double**)realloc(A->Values,new_size*sizeof(double*));
	printf("val %f index %d\n",A->Values[0][0],0); 
	if (new_size>old_size){
		for (i=old_size;i<new_size;i++){
			A->Len[i] = 1;
			A->Indices[i] = (int*)malloc(sizeof(int));
			A->Indices[i][0] = i;
			A->Values[i] = (double*)malloc(sizeof(double));
			A->Values[i][0] = 0;
			printf("val %f index %d\n",A->Values[0][0],0); 
			//printf("val %f index %d\n",T[0],i); 
		}
		
	}
	printf("fertig");
}*/

sparse_matrix* get_resized_matrix(sparse_matrix* A,int mesh_size,int new_size,int equ_index,int var_index){
	int i,j,k,l,r,s;
	int n = mesh_size;
	int m = A->size;
	sparse_matrix* Res = sparse_zero(new_size);
	for (i=0;i<m;i++){
		r = i / n;
		s = i % n;
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j] / n;
			l = A->Indices[i][j] % n;
			if ((equ_index+r)*n+s<new_size && (var_index+k)*n+l<new_size){
				insert_sparse(Res,A->Values[i][j],(equ_index+r)*n+s,(var_index+k)*n+l);
			}
		}
	}
	return Res;
}

void enlarge_matrix(sparse_matrix* A,int mesh_size,int new_size,int equ_index,int var_index){
	int i,j,k,l;
	int old_size = A->size;
	A->Len = (int*)realloc(A->Len,new_size*sizeof(int));
	A->Indices = (int**)realloc(A->Indices,new_size*sizeof(int*));
	A->Values = (double**)realloc(A->Values,new_size*sizeof(double*));
	A->size = new_size;
	for (i=0;i<old_size;i++){
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j] % mesh_size;
			l = A->Indices[i][j] / mesh_size;
			A->Indices[i][j] = (l+var_index)*mesh_size+k;			
		}
	}
	for (i=old_size-1;i>=0;i--){
		k = i % mesh_size;
		l = i / mesh_size;
		A->Indices[(l+equ_index)*mesh_size+k] = A->Indices[i];
		A->Values[(l+equ_index)*mesh_size+k] = A->Values[i];
		A->Len[(l+equ_index)*mesh_size+k] = A->Len[i];
	}
	for (i=0;i<new_size;i++) if (i<equ_index*mesh_size || i>=equ_index*mesh_size+old_size){
		A->Len[i] = 1;
		A->Indices[i] = (int*)malloc(sizeof(int));
		A->Values[i] = (double*)malloc(sizeof(double));
		A->Indices[i][0] = i;
		A->Values[i][0] = 0;
	}
}

double get_mean(double* List,int size){
	int i;
	double sum = 0;
	for (i=0;i<size;i++) sum += List[i];
	sum /= (double)size;
	return sum;
}

double get_partial_mean(double* List,int start,int end){
	int i;
	double sum = 0;
	for (i=start;i<end;i++) sum += List[i];
	sum /= (double)(end-start);
	return sum;
}


double get_sqr_mean(double* List,int size){
	int i;
	double mean = get_mean(List,size);
	double sum = 0;
	for (i=0;i<size;i++) sum += (List[i]-mean)*(List[i]-mean);
	sum /= (double)size;
	return sqrt(sum);
}

sparse_matrix* get_1D_Laplace(double* F,int n){ //gibt A und F zu u''=f
	int i;
	double h = 1./(n-1);
	sparse_matrix* A = sparse_zero(n);
	for (i=1;i<n-1;i++){
		insert_sparse(A,2.,i,i);
		if (i!=1) insert_sparse(A,-1.,i,i-1);
		if (i!=n-2) insert_sparse(A,-1.,i,i+1);
		F[i] *= -h*h;
	}
	insert_sparse(A,1.,0,0);
	insert_sparse(A,1.,n-1,n-1);
	F[0] = 0;
	F[n-1] = 0;
	return A;
}

double sparse_trace(sparse_matrix* A){
	int i,j,J;
	double res = 0;
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++){
			J = A->Indices[i][j];
			if (J==i){
				res += A->Values[i][j];
				break;
			}
		}
	}
	return res;
}

int is_U_triagonal(double** Q,int n){
	int i,j;
	int res = 1;
	for (i=0;i<n;i++){
		for (j=0;j<i;j++) if (Q[i][j]!=0){
			res = 0;
			break;
		}
		if (res==0) break;
	}
	return res;
}

int is_L_triagonal(double** Q,int n){
	int i,j;
	int res = 1;
	for (i=0;i<n;i++){
		for (j=i+1;j<n;j++) if (Q[i][j]!=0){
			res = 0;
			break;
		}
		if (res==0) break;
	}
	return res;
}

double get_det(double** Q,int n){
	int i;
	double det = 1;
	if (n==1){
		return Q[0][0];
	}
	else{
		if (is_U_triagonal(Q,n) || is_L_triagonal(Q,n)){
			for (i=0;i<n;i++) det *= Q[i][i];
			return det;
		}
		else{
			double** L = (double**)malloc(n*sizeof(double*));
			double** U = (double**)malloc(n*sizeof(double*));
			LU_factorzation(Q,L,U,n);
			for (i=0;i<n;i++) det *= L[i][i]*U[i][i];
			for (i=0;i<n;i++){
				free(L[i]);
				free(U[i]);
			}
			free(L);
			free(U);
			return det;
		}
	}
}

double** get_sub_matrix(double** Q,int n,int i,int j){
	int k,l,g,h;
	double** S = (double**)malloc((n-1)*sizeof(double*));
	for (k=0;k<n-1;k++){
		S[k] = (double*)malloc((n-1)*sizeof(double));
		if (k>=i) g = k+1; else g = k;
		for (l=0;l<n-1;l++){
			if (l>=j) h = l+1; else h = l;
			S[k][l] = Q[g][h];
		}
	}
	return S;
}

double** get_inverse(double** Q,int n){
	double** Sub;
	double** Inverse = (double**)malloc(n*sizeof(double*));
	int i,j,k;
	double sign;
	double det = get_det(Q,n);
	for (i=0;i<n;i++){
		if (i % 2 == 0) sign = 1.; else sign = -1.;
		Inverse[i] = (double*)malloc(n*sizeof(double));
		if (n>1) for (j=0;j<n;j++){
			Sub = get_sub_matrix(Q,n,j,i);
			Inverse[i][j] = sign*get_det(Sub,n-1)/det;
			sign *= -1.;
			for (k=0;k<n-1;k++) free(Sub[k]);
			free(Sub);
		}
		else Inverse[0][0] = 1./Q[0][0];
	}
	return Inverse;
}

sparse_matrix* Lap_2D_Dirichlet(double a,double b,int dom_size){
	int i,j,k,l;
	int N = dom_size*dom_size;
	sparse_matrix* A = sparse_zero(N);
	for (i=1;i<dom_size-1;i++){
		for (j=1;j<dom_size-1;j++){
			k = i*dom_size+j;
			insert_sparse(A,2.*(a+b),k,k);
			l = (i+1)*dom_size+j;
			insert_sparse(A,-a,k,l);
			l = (i-1)*dom_size+j;
			insert_sparse(A,-a,k,l);
			l = i*dom_size+j+1;
			insert_sparse(A,-b,k,l);
			l = i*dom_size+j-1;
			insert_sparse(A,-b,k,l);
		}
	}
	for (i=0;i<dom_size;i++){
		k = i*dom_size;
		insert_sparse(A,1.,k,k);
		k = i*dom_size+dom_size-1;
		insert_sparse(A,1.,k,k);
		k = i;
		insert_sparse(A,1.,k,k);
		k = (dom_size-1)*dom_size+i;
		insert_sparse(A,1.,k,k);
	}
	return A;
}

double test_sparse_LU(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	sparse_matrix* P = sparse_product(L,U);
	sparse_add(P,A,-1.);
	double res = matrix_norm(P)/matrix_norm(A);
	free_sparse(P);
	return res;
}

sparse_matrix* get_sub_sparse_matrix(sparse_matrix* A,int start,int end,int* Map,int* Inv_Map){
	int i,j,k;
	double a;
	sparse_matrix* Sub = sparse_zero(end-start);
	for (i=start;i<end;i++){
		for (k=0;k<A->Len[i];k++){
			j = A->Indices[i][k];
			a = A->Values[i][k];
			if (a!=0 && j>=start && j<end) insert_sparse(Sub,a,i-start,j-start);
		}
		if (Map!=NULL) Map[i] = i-start;
		if (Inv_Map!=NULL) Inv_Map[i-start] = i;
	}
	return Sub;
}

void get_sub_cross_couplings(sparse_matrix* A,int last,int start,int end,
double** Lower,double** Upper){ // Arrays mit Nullen initialisieren  !
	int i,j,k;
	for (i=start;i<end;i++){
		for (k=0;k<A->Len[i];k++){
			j = A->Indices[i][k];
			if (j>=last && j<start) Lower[i-start][j-last] = A->Values[i][k];  // Achtung: nur rückwärtige Kopplung enthalten !
		}
	}
	for (i=last;i<start;i++){
		for (k=0;k<A->Len[i];k++){
			j = A->Indices[i][k];
			if (j>=start && j<end) Upper[j-start][i-last] = A->Values[i][k];  // Achtung: nur rückwärtige Kopplung enthalten !
		}
	}
}

int sparse_rb_comp_ind(const void* A,const void* B){
	int* a = (int*)A;
	int* b = (int*)B;
	if ((*a)<(*b)) return 1; else return 0;
}

void sparse_rb_print_key(const void* A){
	int* a = (int*)A;
	printf("%d  |",*a);
}

void sparse_rb_print_info(void* I){
	int* i = (int*)I;
	printf("%d\n",*i);
}

void sparse_rb_free_key(void* I){
	int* i = (int*)I;
	free(i);
}

void sparse_rb_free_info(void* I){
	int* i = (int*)I;
	free(i);
}

int sparse_is_isolated(sparse_matrix* A,int i){
	int j,k;
	double a;
	int res = 1;
	for (j=0;j<A->Len[i];j++){
		k = A->Indices[i][j];
		a = A->Values[i][j];
		if (k!=i && a!=0) res = 0;
	}
	return res;
}

int get_smallest_connected_index(sparse_matrix* A,int* Map){
	int i;
	int res = -1;
	for (i=0;i<A->size;i++){
		if (Map[i]<0 && !sparse_is_isolated(A,i)) {
			res = i;
			break;
		}
	}
	return res;
}

void sparse_get_optimized_index_map(sparse_matrix* A,int* Map,int* Inv_Map){
	int* Index;
	int* Value;
	rb_red_blk_node* Smallest;
	int i,j,k,Ind;
	int n = A->size;
	for (i=0;i<n;i++) Map[i] = -1;
	rb_red_blk_tree* Index_tree = RBTreeCreate(&sparse_rb_comp_ind,&sparse_rb_free_key,
	 &sparse_rb_free_info,&sparse_rb_print_key,&sparse_rb_print_info);
	k = get_smallest_connected_index(A,Map);
	i = 0;
	while(k>=0){
		Index = (int*)malloc(sizeof(int));
		Value = (int*)malloc(sizeof(int));
		*Index = k;
		*Value = i;
		Map[k] = i;
		Inv_Map[i] = k;
		RBTreeInsert(Index_tree,Value,Index);
		Smallest = RBLargest(Index_tree);
		i++;
		do{
			Ind = *(int*)Smallest->info;
			for (j=0;j<A->Len[Ind];j++){
				k = A->Indices[Ind][j];
				if (Map[k]<0){
					Index = (int*)malloc(sizeof(int));
					Value = (int*)malloc(sizeof(int));
					*Index = k;
					*Value = i;
					RBTreeInsert(Index_tree,Value,Index);
					Map[k] = i;
					Inv_Map[i] = k;
					i++;
				}
			}
			RBDelete(Index_tree,Smallest);
			Smallest = RBLargest(Index_tree);
		}while(Smallest!=NULL);
		k = get_smallest_connected_index(A,Map);
	};
	RBTreeDestroy(Index_tree);
	for (k=0;k<n;k++) if (Map[k]<0){
		Map[k] = i;
		Inv_Map[i] = k;
		i++;
	}
}

int get_new_block_index(sparse_matrix* A,sparse_matrix* TrA,int old_pos,int cur_pos){
	int i,k,new_pos;
	int imax = cur_pos;
	for (i=old_pos;i<cur_pos;i++) if (A->Len[i]>0){
		k = A->Indices[i][A->Len[i]-1];
		if (k>imax) imax = k;
	}
	int jmax = cur_pos;
	for (i=old_pos;i<cur_pos;i++) if (TrA->Len[i]>0){
		k = TrA->Indices[i][TrA->Len[i]-1];
		if (k>jmax) jmax = k;
	}
	new_pos = imax > jmax ? imax+1 : jmax+1;
	if (new_pos>A->size) new_pos = -1;
	return new_pos;
}

void LU_sparse_block_factorization(sparse_matrix* A,sparse_matrix* LB,
 sparse_matrix* UB,int* Map,int* Inv_Map){
	sparse_matrix* Sub;
	sparse_matrix* L_Sub;
	sparse_matrix* U_Sub;
	sparse_matrix* TrU_Sub;
	double** Cross_L;
	double** Cross_U;
	int i,j,new_pos,cur_pos,old_pos;
	int finished = 0;
	int n =A->size;
	int w0 = get_ave_band_width(A);
	sparse_matrix* B = sparse_zero(n);
	sparse_get_optimized_index_map(A,Map,Inv_Map);
	map_indices(A,B,NULL,NULL,Map);
	sparse_matrix* TrB = get_transpose(B,n);
	int w = get_ave_band_width(B);
	printf("matrix size: %d\n",A->size);
	printf("bandwidth reduction: %f\n",(double)w/w0);
	old_pos = 0;
	cur_pos = w/2;
	new_pos = get_new_block_index(B,TrB,old_pos,cur_pos);
	do{
		// bestimme LU von old_pos-cur_pos
		Sub = get_sub_sparse_matrix(B,old_pos,cur_pos,NULL,NULL);
		L_Sub = sparse_zero(cur_pos-old_pos);
		U_Sub = sparse_zero(cur_pos-old_pos);
		complete_LU_factorization(Sub,L_Sub,U_Sub);
		sparse_in_sparse(LB,L_Sub,old_pos,old_pos);
		sparse_in_sparse(UB,U_Sub,old_pos,old_pos);
		if (new_pos>=0){
			// bestimme Kreuzterme durch triang_invert 
			TrU_Sub = get_transpose(U_Sub,U_Sub->size);
			Cross_L = zero_matrix(new_pos-cur_pos,cur_pos-old_pos);
			Cross_U = zero_matrix(new_pos-cur_pos,cur_pos-old_pos);
			get_sub_cross_couplings(B,old_pos,cur_pos,new_pos,Cross_L,Cross_U);
			for (i=0;i<new_pos-cur_pos;i++){
				L_triang_invert(L_Sub,Cross_U[i]);
				L_triang_invert(TrU_Sub,Cross_L[i]);
				for (j=0;j<cur_pos-old_pos;j++){
					if (Cross_L[i][j]!=0) set_sparse(LB,Cross_L[i][j],i+cur_pos,j+old_pos);
					if (Cross_U[i][j]!=0) set_sparse(UB,Cross_U[i][j],j+old_pos,i+cur_pos);
				}
			}
			// update b von cur_pos-new_pos
			for (i=0;i<new_pos-cur_pos;i++){
				for (j=0;j<new_pos-cur_pos;j++) insert_sparse(B,-scalar(Cross_L[i],Cross_U[j],
				 cur_pos-old_pos),i+cur_pos,j+cur_pos);
			}
			// clean
			for (i=0;i<new_pos-cur_pos;i++){
				free(Cross_L[i]);
				free(Cross_U[i]);
			}
			free(Cross_L);
			free(Cross_U);
			free_sparse(TrU_Sub);
		}else finished = 1;
		free_sparse(Sub);
		free_sparse(L_Sub);
		free_sparse(U_Sub);
		
		if (cur_pos-old_pos>1) printf("block %d to %d finished\n",old_pos,cur_pos);
		// new block
		old_pos = cur_pos;
		cur_pos = new_pos;
		new_pos = get_new_block_index(B,TrB,old_pos,cur_pos);
	}while(!finished);
	
	free_sparse(TrB);
	free_sparse(B);
 }

int is_L_triang_matrix(sparse_matrix* A){
	int i;
	int n = A->size;
	for (i=0;i<n;i++){
		if (A->Indices[i][A->Len[i]-1]>i && A->Values[i][A->Len[i]-1]!=0) return 0;
	}
	return 1;
}

int is_U_triang_matrix(sparse_matrix* A){
	int i;
	int n = A->size;
	for (i=0;i<n;i++){
		if (A->Indices[i][0]<i && A->Values[i][0]!=0) return 0;
	}
	return 1;
}

void L_block_triang_invert(sparse_matrix* L,int* Map,int* Inv_Map,double* X){
	if (Map==NULL || Inv_Map==NULL) L_triang_invert(L,X);
	else{
		int n = L->size;
		sparse_matrix* T = sparse_zero(n);
		vector_permutation(X,Inv_Map,n);
		map_indices(L,T,NULL,NULL,Map);
		L_triang_invert(T,X);
		vector_permutation(X,Map,n);
		free_sparse(T);
	}
}

void U_block_triang_invert(sparse_matrix* U,int* Map,int* Inv_Map,double* X){
	if (Map==NULL || Inv_Map==NULL) U_triang_invert(U,X);
	else{
		int n = U->size;
		sparse_matrix* T = sparse_zero(n);
		vector_permutation(X,Inv_Map,n);
		map_indices(U,T,NULL,NULL,Map);
		U_triang_invert(T,X);
		vector_permutation(X,Map,n);
		free_sparse(T);
	}
}

// begin tree_matrix

tree_sparse_matrix init_tree_sparse(tree_sparse_matrix* Parent,int size){
	tree_sparse_matrix Res;
	Res.Parent = Parent;
	Res.map_size = size;
	Res.Map = zero_int_list(size);
	Res.Inverse_Map = zero_int_list(size);
	Res.Childs = (tree_sparse_matrix**)malloc(sizeof(tree_sparse_matrix*));
	Res.child_number = 0;
	Res.Lower_Cross = NULL;
	Res.Upper_Cross = NULL;
	Res.Leaf = NULL;
	return Res;
}

// free procedure nicht vergessen !

void invert_L_tree_matrix(tree_sparse_matrix* L,double* X){
	int i,sub_size,start,end;
	double* Sub_X;
	int n = L->map_size;
	if (L->Leaf!=NULL) L_triang_invert(L->Leaf,X);
	else{
		vector_permutation(X,L->Inverse_Map,n);
		start = 0;
		end = 0;
		for (i=0;i<L->child_number;i++){
			sub_size = (*(L->Childs))[i].map_size;
			end += sub_size;
			Sub_X = restrict_vector_by(start,end,X);	// beginnt mit index 0
			linear_map(Sub_X,-1.,L->Lower_Cross,X);     // Lower_Cross bei index 0 beginnnend
			invert_L_tree_matrix(L->Childs[i],Sub_X);	// L_Child auch 
			copy_vector_content(Sub_X,X,0,start,sub_size);
			start += sub_size;
			free(Sub_X);
		}
		vector_permutation(X,L->Map,n);
	}
}

// end tree_matrix

/*void Multi_LU_sparse_block_factorization(sparse_matrix* A,tree_sparse_matrix* L,
 tree_sparse_matrix* U,int crit_size){
	sparse_matrix* Sub;
	sparse_matrix* L_Sub;
	sparse_matrix* U_Sub;
	sparse_matrix* TrU_Sub;
	sparse_matrix* T;
	int* Sub_Map = NULL;
	int* Sub_Inv_Map = NULL;
	double** Cross_L;
	double** Cross_U;
	int i,j,new_pos,cur_pos,old_pos;
	int finished = 0;
	int n =A->size;
	int w0 = get_ave_band_width(A);
	
	sparse_matrix* LB = sparse_zero(n);
	sparse_matrix* UB = sparse_zero(n);
	sparse_matrix* B = sparse_zero(n);
	int* Map = zero_int_list(size);
	int* Inv_Map = zero_int_list(size);
	sparse_get_optimized_index_map(A,Map,Inv_Map);
	L->Map = Map;
	L->Inverse_Map = Inv_Map;
	map_indices(A,B,NULL,NULL,Map);
	sparse_matrix* TrB = get_transpose(B);
	int w = get_ave_band_width(B);
	printf("bandwidth reduction: %f\n",(double)w/w0);
	old_pos = 0;
	cur_pos = w/2;
	new_pos = get_new_block_index(B,TrB,old_pos,cur_pos);
	do{
		// bestimme LU von old_pos-cur_pos
		Sub = get_sub_sparse_matrix(B,old_pos,cur_pos,NULL,NULL);
		L_Sub = sparse_zero(cur_pos-old_pos);
		U_Sub = sparse_zero(cur_pos-old_pos);
		//if (old_pos>0 && cur_pos-old_pos>crit_size){
		if (cur_pos-old_pos>1 && crit_size>0){
			Sub_Map = zero_int_list(cur_pos-old_pos);
			Sub_Inv_Map = zero_int_list(cur_pos-old_pos);
			Multi_LU_sparse_block_factorization(Sub,L_Sub,U_Sub,Sub_Map,Sub_Inv_Map,crit_size-1);
		}
		else{
			Sub_Map = NULL;
			Sub_Inv_Map = NULL;
			complete_LU_factorization(Sub,L_Sub,U_Sub);
		}
		
			printf("level %d L triangular: %d\n",2-crit_size,is_L_triang_matrix(L_Sub));
			printf("level %d U triangular: %d\n",2-crit_size,is_U_triang_matrix(U_Sub));
		
		
		sparse_in_sparse(LB,L_Sub,old_pos,old_pos);
		sparse_in_sparse(UB,U_Sub,old_pos,old_pos);
			if (new_pos>=0){
			// bestimme Kreuzterme durch triang_invert 
			TrU_Sub = get_transpose(U_Sub);
			Cross_L = zero_matrix(new_pos-cur_pos,cur_pos-old_pos);
			Cross_U = zero_matrix(new_pos-cur_pos,cur_pos-old_pos);
			get_sub_cross_couplings(B,old_pos,cur_pos,new_pos,Cross_L,Cross_U);
			for (i=0;i<new_pos-cur_pos;i++){
				L_block_triang_invert(L_Sub,Sub_Map,Sub_Inv_Map,Cross_U[i]);
				L_block_triang_invert(TrU_Sub,Sub_Map,Sub_Inv_Map,Cross_L[i]);
				for (j=0;j<cur_pos-old_pos;j++){
					if (Cross_L[i][j]!=0) set_sparse(LB,Cross_L[i][j],i+cur_pos,j+old_pos);
					if (Cross_U[i][j]!=0) set_sparse(UB,Cross_U[i][j],j+old_pos,i+cur_pos);
				}
			}
			// update b von cur_pos-new_pos
			T = get_sub_sparse_matrix(B,cur_pos,new_pos,NULL,NULL);
			free_sparse(T);
			
			for (i=0;i<new_pos-cur_pos;i++){
				for (j=0;j<new_pos-cur_pos;j++) insert_sparse(B,-scalar(Cross_L[i],Cross_U[j],
				 cur_pos-old_pos),i+cur_pos,j+cur_pos);
			}
			// clean
			for (i=0;i<new_pos-cur_pos;i++){
				free(Cross_L[i]);
				free(Cross_U[i]);
			}
			free(Cross_L);
			free(Cross_U);
			free_sparse(TrU_Sub);
		}else finished = 1;
		free_sparse(Sub);
		free_sparse(L_Sub);
		free_sparse(U_Sub);
		if (Sub_Map!=NULL){
			free(Sub_Map);
			free(Sub_Inv_Map);
		}
		if (cur_pos-old_pos>1){
			printf("level: %d block %d to %d finished\n",2-crit_size,old_pos,cur_pos);
		}
		
		// new block
		old_pos = cur_pos;
		cur_pos = new_pos;
		new_pos = get_new_block_index(B,TrB,old_pos,cur_pos);
	}while(!finished);
		
	map_indices(LB,L,NULL,NULL,Inv_Map);
	map_indices(UB,U,NULL,NULL,Inv_Map);
	
	printf("LU deviation level %d: %f\n",2-crit_size,test_sparse_LU(A,L,U));

	
	free_sparse(TrB);
	free_sparse(B);
	free_sparse(LB);
	free_sparse(UB);
}*/

void LU_sparse_block_solver(sparse_matrix* L,sparse_matrix* U,int* Map,int* Inv_Map,
 double* X,double* B){
	int n = L->size;
	copy_vector_content(B,X,0,0,n);
	vector_permutation(X,Inv_Map,n);
	L_triang_invert(L,X);
	U_triang_invert(U,X);
	vector_permutation(X,Map,n);
}

int power_method(sparse_matrix* A,double* max_eig,double* Max_vec,int max_iter,double eps,sparse_matrix* Original,double eps_neg){			// get largest eigenvalue and eigenvector
	int i;
	int n = A->size;			// Max_vec normiert 
	double prev_eig,s,r;
	double q = matrix_norm(A);
	double* Prev_vec = zero_vector(n);
	i = 0;
	do{
		prev_eig = *max_eig;
		copy_vector_to(Max_vec,Prev_vec,n);
		sparse_multiplication(A,Max_vec);
		*max_eig = scalar(Max_vec,Prev_vec,n);
		s = 1./euklid_norm(Max_vec,n);
		scalar_mult(s,Max_vec,n);
		r = fabs(*max_eig-prev_eig)/q;
		//if (Original!=NULL && sparse_bilinear(Max_vec,Original,Max_vec)<-eps_neg) break;
		i++;
	}while(r>eps && i<max_iter);
	free(Prev_vec);
	if (i==max_iter) return 0; else return i;
}

void set_random_normal_vector(double* X,int n){
	int i;
	srand(time(NULL));
	for (i=0;i<n;i++) X[i] = (double)rand()/RAND_MAX;
	vector_normalize(X,n);
}

int get_most_negative_eigenvector(sparse_matrix* A,double* eigenvalue, double* Eigenvector,int max_iter,double eps){
	const int pre_iter = 100;
	int i,iter;
	double shift;
	int n = A->size;
	set_random_normal_vector(Eigenvector,n);
	iter = power_method(A,&shift,Eigenvector,pre_iter,eps,NULL,0);	
	//printf("estimated norm: %f\n",shift);
	//printf("pre-iterations: %d\n",iter);
	sparse_matrix* B = clone(A);
	for (i=0;i<n;i++) insert_sparse(B,-shift,i,i);
	set_random_normal_vector(Eigenvector,n);
	iter = power_method(B,eigenvalue,Eigenvector,max_iter,eps,NULL,0);
	*eigenvalue += shift;
	free_sparse(B);
	return iter;
}

int get_neg_curvature_direction(sparse_matrix* A,double* curvature, double* Direction,int max_iter,double eps,double neg_eps){
	const int pre_iter = 100;
	int i,iter;
	double shift;
	int n = A->size;
	set_random_normal_vector(Direction,n);
	iter = power_method(A,&shift,Direction,pre_iter,eps,NULL,0);
	//printf("estimated norm: %f\n",shift);
	//printf("pre-iterations: %d\n",iter);
	sparse_matrix* B = clone(A);
	for (i=0;i<n;i++) insert_sparse(B,-shift,i,i);
	set_random_normal_vector(Direction,n);
	iter = power_method(B,curvature,Direction,max_iter,eps,A,neg_eps);
	*curvature += shift;
	free_sparse(B);
	return iter;
}

/*int row_overlap(sparse_matrix* A,sparse_matrix* B,int i,int j){
	if ((B->Indices[j][B->Len[j]-1])<(A->Indices[i][0]) || (B->Indices[j][B->Len[0]])>(A->Indices[i][A->Len[i]-1])) return 0;
	else return 1;
}*/

double** get_row_pairs(sparse_matrix* A,sparse_matrix* BT,int i,int j,int* Size){
	int c,k,l,r,s;
	*Size = 2*(A->Len[i]);
	double** Pairs = (double**)malloc((*Size)*sizeof(double*));
	k = 0;
	c = 0;
	s = BT->Indices[j][k];
	for (l=0;l<A->Len[i];l++){
		r = A->Indices[i][l];		
		while(s<r && k<BT->Len[j]){
			k++;
			s = BT->Indices[j][k];
		}
		if (k==BT->Len[j]) break;
		if (s==r){
			Pairs[c] = &(A->Values[i][l]);
			Pairs[c+1] = &(BT->Values[j][k]);
			c += 2;
		}
	}
	*Size = c;
	Pairs = (double**)realloc(Pairs,c*sizeof(double*));
	return Pairs;
}

product_pattern set_product_pattern(sparse_matrix* A,sparse_matrix* B,sparse_matrix* B_transposed,sparse_matrix** AB){
	int i,j,J;																	// if B symmetric B_transposed -> NULL
	int row_size = 0;	
	product_pattern pattern;
	sparse_matrix* BT;
	sparse_matrix* Fill_pattern;
	if (B==NULL && B_transposed!=NULL){
		int n = get_max_col_index(B_transposed)+1;
		sparse_matrix* H = get_transpose(B_transposed,n);
		Fill_pattern = sparse_product(A,H);		
		free_sparse(H);
	}
	else Fill_pattern = sparse_product(A,B);
	//sparse_approximate(Fill_pattern,tol);
	if (AB!=NULL) *AB = Fill_pattern;
	if (B_transposed==NULL) BT = B;else BT = B_transposed;
	
	int n = Fill_pattern->size;
	pattern.Pairs = (double****)malloc(n*sizeof(double***));
	pattern.Indices = (int**)malloc(n*sizeof(int*));
	pattern.Size_2 = (int**)malloc(n*sizeof(int*));
	pattern.Size_1 = (int*)malloc(n*sizeof(int));
	pattern.Size_0 = n;
	
	for (i=0;i<Fill_pattern->size;i++){
		pattern.Pairs[i] = (double***)malloc(Fill_pattern->Len[i]*sizeof(double**));
		pattern.Indices[i] = (int*)malloc(Fill_pattern->Len[i]*sizeof(int));
		pattern.Size_2[i] = (int*)malloc(Fill_pattern->Len[i]*sizeof(int));
		pattern.Size_1[i] = Fill_pattern->Len[i];
		for (j=0;j<Fill_pattern->Len[i];j++){	
			J = Fill_pattern->Indices[i][j];		
			pattern.Indices[i][j] = J;
			pattern.Pairs[i][j] = get_row_pairs(A,BT,i,J,&row_size);			
			pattern.Size_2[i][j] = row_size;
		}																	
	}
	if (AB==NULL){free_sparse(Fill_pattern);}
	return pattern;
}

void free_product_pattern(product_pattern* P){
	int j,k;
	if (P!=NULL);
	for (j=0;j<P->Size_0;j++){
		for (k=0;k<P->Size_1[j];k++){
			free(P->Pairs[j][k]);
		}
		free(P->Pairs[j]);
		free(P->Indices[j]);
		free(P->Size_2[j]);
	}
	free(P->Pairs);
	free(P->Indices);
	free(P->Size_1);	
}

void fast_sparse_product(sparse_matrix* P, sparse_matrix* A,sparse_matrix* BT,product_pattern* Pattern){  		// P = A.B
	int i,j,k;																									// BT must be transposed of B
	double sum;																									// A,BT must stay at the smae locations in memory ! 
	double* Addr1;
	double* Addr2;
	int n = Pattern->Size_0;
	
	/*sparse_matrix* P = malloc(sizeof(sparse_matrix));
	P->size = n;
	P->Len = (int*)malloc(n*sizeof(int));
	P->Indices = (int**)malloc(n*sizeof(int*));
	P->Values = (double**)malloc(n*sizeof(double*));*/
	
//#ifdef _OPENMP	
	const int chunk_size = 10000;
	
#line 4219 "linear_algebra.c"
#pragma omp parallel private(i,j,k,sum,Addr1,Addr2)
{
#line 4221 "linear_algebra.c"
	#pragma omp for schedule(dynamic,chunk_size) 
	for (i=0;i<n;i++){		
		
		/*P->Len[i] = Pattern->Size_1[i];
		P->Indices[i] = (int*)malloc(P->Len[i]*sizeof(int));
		P->Values[i] = (double*)malloc(P->Len[i]*sizeof(double));*/
		/*if (Pattern->Size_1[i]!=P->Len[i]){
			printf("error in fast_sparse_product: row length does not match pattern in line %d -> abort\n",i);
			exit(0);
		}*/
		for (j=0;j<Pattern->Size_1[i];j++){
			sum = 0;
			for (k=0;k<Pattern->Size_2[i][j];k += 2){
				Addr1 = Pattern->Pairs[i][j][k];
				Addr2 = Pattern->Pairs[i][j][k+1];
				sum += (*Addr1)*(*Addr2);
			}
			P->Indices[i][j] = Pattern->Indices[i][j];
			P->Values[i][j] = sum;
		}
	}
  		
}
//#endif

}
