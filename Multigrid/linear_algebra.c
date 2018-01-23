
#include "linear_algebra.h"

// global variables 

static int lg_thread_num = 1;

// Code ///////////////////////////////////////////////

void set_thread_num(int n){
	#ifdef _OPENMP
	if (n==AUTOMATIC){
		lg_thread_num = omp_get_num_procs();
		printf("using %d processors\n",lg_thread_num);
	} 
	else lg_thread_num = n;
	omp_set_num_threads(lg_thread_num);
	#endif 
}

double* clone_vector(double* x,int n){
	int i;
	double* res = (double*)malloc(n*sizeof(double));
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i=0;i<n;i++){res[i] = x[i];}
	return res;
}

void copy_vector_to(double* x,double* y,int n){
	int i;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i=0;i<n;i++) y[i] = x[i];
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
	#ifdef _OPENMP
	#pragma omp parallel for private(j,sum)
	#endif
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
	#pragma omp parallel for private(j,sum)
	#endif
	for (i=0;i<n;i++){
		sum = 0; 
		for (j=0;j<(A->Len[i]);j++){sum += A->Values[i][j]*y[A->Indices[i][j]];}
		 x[i] += fac*sum;
	}	
}

void sparse_multiplication(sparse_matrix* A,double* x){
	int i,j;
	double sum;
	int n = A->size;
	double* y = (double*)malloc(n*sizeof(double));
	#ifdef _OPENMP
	#pragma omp parallel for private(j,sum)
	#endif
	for (i=0;i<n;i++){
		sum = 0;
		for (j=0;j<(A->Len[i]);j++){sum += A->Values[i][j]*x[A->Indices[i][j]];}
		y[i] = sum;
	}
	copy_vector_to(y,x,n);
	free(y);
}

sparse_matrix* sparse_product(sparse_matrix* A,sparse_matrix* B){ //P = A*B
	int i,j,k,ind_k;
	double a;
	int n = A->size;
	sparse_matrix* P = sparse_zero(n);
	#ifdef _OPENMP
	#pragma omp parallel for private(j,k,ind_k,a)
	#endif
	for (i=0;i<n;i++){
		for (k=0;k<A->Len[i];k++){
			ind_k = A->Indices[i][k];
			a = A->Values[i][k];
			for (j=0;j<B->Len[ind_k];j++) insert_sparse(P,a*(B->Values[ind_k][j]),i,B->Indices[ind_k][j]);
		}
	}
	//delete_zeros(P);
	return P;
}

double scalar(double* x1,double* x2,int n){
	int i;
	double res = 0;
	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:res)
	#endif
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
	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:sum)
	#endif
	for (i=0;i<n;i++){
		d = x1[i]-x2[i];
		sum = sum + d*d;
	}
	return sqrt(sum);
}

void scalar_mult(double a,double* x,int n){
	int i;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
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
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i=0;i<n;i++){
		x[i] += factor*y[i];
	}
}

void scalar_sparse_mult(double a,sparse_matrix* A){
	int i,j;
	#ifdef _OPENMP
	#pragma omp parallel for private(j)
	#endif
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
	res->Len = malloc(n*sizeof(int));
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
	res->Len = malloc(n*sizeof(int));
	for (i=0;i<n;i++){
		res->Values[i] = NULL;
		res->Indices1[i] = NULL;
		res->Indices2[i] = NULL;
		res->Len[i] = 0;
	}
	res->size = n;
	return res;
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
	int i,j;
	for (i=0;i<A->size;i++){		
		if (A->Len[i]>1){
			j = 0;
			do{
				if (A->Values[i][j]==0){remove_element_at(A,i,j);}
				else{j++;}
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

void print_sparse(sparse_matrix* A){
	int i,j,n;
	double sum;
	if (A!=NULL){
		FILE* file = fopen("/home/radszuweit/Daten/sparse_matrix","w");
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
		FILE* file = fopen("/home/radszuweit/Daten/sparse_table","w");
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
		FILE* file = fopen("/home/radszuweit/Daten/vector","w");
		for (i=0;i<n;i++){
			fprintf(file,"%f\n",V[i]);
		}
		fclose(file);
	}
}

void print_table(double* V,char* name,int n){
	int i;
	char dir[512];
	sprintf(dir,"/home/radszuweit/Daten/%s",name);
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
		FILE* file = fopen("/home/radszuweit/Daten/list","w");
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
	A->Values[i] = (double*)realloc(A->Values[i],(oldsize+1)*sizeof(double));
	A->Indices[i] = (int*)realloc(A->Indices[i],(oldsize+1)*sizeof(int));
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
	A->Values[i] = malloc(sizeof(double));
	A->Values[i][0] = 0;
	A->Indices[i] = malloc(sizeof(int));
	A->Indices[i][0] = i;
	A->Len[i] = 1;
}

sparse_matrix* get_transpose(sparse_matrix* A){
	int i,j,r;
	int n = A->size;
	sparse_matrix* Res = sparse_zero(n);
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

void sparse_approximate(sparse_matrix* A,double tol){
	int i,j,k;
	double a,max;
	int n = A->size;
	for (i=0;i<n;i++){
		max = 0;
		for (j=0;j<A->Len[i];j++){
			a = fabs(A->Values[i][j]);
			if (A->Indices[i][j]!= i && a>max) max = a;
		}
		j = 0;
		while (j<A->Len[i]){
			k = A->Indices[i][j];
			if (k!=i && fabs(A->Values[i][j])<max*tol) remove_element_at(A,i,j);
			else j++;
		}
	}
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

void copy_sparse_content(sparse_matrix* A,sparse_matrix* B){
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
		FILE* file = fopen("/home/radszuweit/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=0;j<m;j++){
				fprintf(file,"%f   ",A[i][j]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
}

void print_2D_array(double** A,int n){
	int i,j;
	if (A!=NULL){
		FILE* file = fopen("/home/radszuweit/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=1;j<=A[i][0];j++){
				fprintf(file,"%f   ",A[i][j]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
}

void print_2D_list(int** A,int n){
	int i,j;
	if (A!=NULL){
		FILE* file = fopen("/home/radszuweit/Daten/array_matrix","w");
		for (i=0;i<n;i++){
			for (j=1;j<=A[i][0];j++){
				fprintf(file,"%d   ",A[i][j]);
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
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

void incomplete_LU_factorization(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	double sum,d;
	int i,j,jr;
	int n = A->size;
	for (i=0;i<n;i++){
		insert_sparse(U,1,i,i);
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
				else printf("ILU: Warning, diagonal element is zero at index %d!\n",i);
				insert_sparse(U,sum,j,i);
			}
		}
	}
	sparse_matrix* TrU = get_transpose(U);
	copy_sparse_content(U,TrU);
	free_sparse(TrU);
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
	int n = L->size;
	double* b = clone_vector(x,n);
	double sum,diag;
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
	int n = U->size;
	double* b = clone_vector(x,n);
	double sum,diag;
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

int get_max_band_width(sparse_matrix* A){
	int i,d;
	int max = 0;
	for (i=0;i<A->size;i++){
		if (A->Len[i]>0) d = A->Indices[i][A->Len[i]-1]-A->Indices[i][0]+1; else d = 0;
		if (d>max) max = d;
	}
	return max;
}

int get_ave_band_width(sparse_matrix* A){
	int i;
	int n = 0;
	int sum = 0;
	for (i=0;i<A->size;i++){
		if (A->Len[i]>0){
			sum += A->Indices[i][A->Len[i]-1]-A->Indices[i][0]+1;
			n++;
		}
	}
	return ceil((double)sum/n);
}

int L_triang_fast_invert(sparse_matrix* L,int** Ind,double** Val,int size,int I,int max_band_width,double tol){
	int i,j,k,K,next,last;
	double a,sum,diag,norm;
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
	double a,sum,diag,norm;
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
	sparse_matrix* Transpose = get_transpose(A);
	for (i=0;i<n;i++) Transpose->Len[i] = L_triang_fast_invert(L,Transpose->Indices,Transpose->Values,Transpose->Len[i],i,L_width,tol);
	sparse_approximate(Transpose,1E-2);
	for (i=0;i<n;i++) Transpose->Len[i] = U_triang_fast_invert(U,Transpose->Indices,Transpose->Values,Transpose->Len[i],i,U_width,tol);
	sparse_approximate(Transpose,1E-4);
	sparse_matrix* Res = get_transpose(Transpose);
	free_sparse(Transpose);
	return Res;
}

double get_diag_dominance(sparse_matrix* A){
	int i,j,J;
	double diag,sum;
	double max = 0;
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
				printf("diagonal element is zero at index %d -> abort Gauss-Seidel\n",i);
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

sparse_matrix* make_positive_diag(sparse_matrix* A){
	int i,j;
	sparse_matrix* Res = sparse_identity(A->size);
	for (i=0;i<A->size;i++) if (get_diag(A,i)<0){
		for (j=0;j<A->Len[i];j++) A->Values[i][j] *= -1.;
		Res->Values[i][0] *= -1.;
	}
	return Res;
}

void simple_pivot(sparse_matrix* A,double* B,int* Map){				// Spaltenmaximumsstrategie
	int i,j,max_i,start;
	double diag_i,diag_j,b,new_diag;
	int n = A->size;
	if (Map!=NULL) for (i=0;i<n;i++) Map[i] = i;
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
		if (i % (n/10) ==0) printf("simple pivot: %d \r",100*i/n);
	}
	printf("simple pivot: completed \n");
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
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			sum += fabs(A[i][j]-B[i][j]);
			sum_A += fabs(A[i][j]);
		}
	}
	printf("LU-deviation: %f\n",sum/sum_A);
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

double* zero_vector(int n){
	int i;
	double* Res = (double*)malloc(n*sizeof(double));
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
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

int* convert_to_UMFPACK_sizes(sparse_matrix* A){
	int i,k;
	int n = A->size;
	int* Res = (int*)malloc((n+1)*sizeof(int));
	k = 0;
	Res[0] = 0;
	for (i=0;i<n;i++){
		k += A->Len[i];
		Res[i+1] = k;
	}
	return Res;
}

int* convert_to_UMFPACK_indices(sparse_matrix* A,int N){
	int i,j,k;
	int n = A->size;
	int* Res = (int*)malloc(N*sizeof(int));
	k = 0;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			Res[k] = A->Indices[i][j];
			k++;
		}
	}
	return Res;
}

double* convert_to_UMFPACK_Values(sparse_matrix* A,int N){
	int i,j,k;
	int n = A->size;
	double* Res = (double*)malloc(N*sizeof(int));
	k = 0;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			Res[k] = A->Values[i][j];
			k++;
		}
	}
	return Res;
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

sparse_matrix* get_test_matrix(){
	sparse_matrix* Res = sparse_identity(2);
	insert_sparse(Res,0.5,0,1);
	insert_sparse(Res,0.5,1,0);
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
		complete_LU_factorization(Sub,L_Sub,U_Sub);
		sparse_in_sparse(LB,L_Sub,old_pos,old_pos);
		sparse_in_sparse(UB,U_Sub,old_pos,old_pos);
		if (new_pos>=0){
			// bestimme Kreuzterme durch triang_invert 
			TrU_Sub = get_transpose(U_Sub);
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
