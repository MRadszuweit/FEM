
#include "Multigrid.h"

// globale Variablen

static double eps_minus = 0.3;
static double eps_plus = 0.5;
static double AMG_tol = 1E-5;

const int thread_number = 1;

//static int iteration_counter;
static double SOR_alpha = 1.;
static int AMG_smoothing_iterations = 2;
static int AMG_aggr_smoothing_iterations = 4;
static int AMG_coarsening = STANDARD;
static int AMG_info = YES;
static int AMG_deg_of_freedom = 1;
static sparse_matrix* AMG_A = NULL;
static sparse_matrix* AMG_L = NULL;
static sparse_matrix* AMG_U = NULL;
//static double* test_vector = NULL;

static int (*F_Prolong)(sparse_matrix* A,sparse_matrix* B,int* Label,int i,
 int** Indices,double** Values,int** Status,int** Strong,int** Positions) = NULL;
static void (*AMG_coarsest_solver)(sparse_matrix* L,sparse_matrix* U,double* F,double* Sol,int size) = NULL;

static int func_counter = 0;

// Funktionen

void PCG_mult(int n,double* x,double* y){
	copy_vector_to(x,y,n);
	if (n==AMG_A->size) sparse_multiplication(AMG_A,y);
	else printf("Warning: wrong matrix in pcg multiplication\n");
}

int PCG_precon(int n,double* R,double* Z){
	copy_vector_to(R,Z,n);
	//for (i=0;i<n;i++) Z[i] /= get_diag(AMG_A,i); 
	//L_triang_invert(AMG_L,Z);
	//U_triang_invert(AMG_U,Z);
	return 1;
}

void AMG_set_PCG_matrices(sparse_matrix* A,sparse_matrix* L,sparse_matrix* U){
	AMG_A = A;
	AMG_L = L;
	AMG_U = U;
}

int get_max_coupled(sparse_matrix* A,int i){
	int j;
	double min = 0;
	int index = -1;
	for (j=0;j<A->Len[i];j++){
		if (A->Indices[i][j]!=i && A->Values[i][j]<min){
			min = A->Values[i][j];
			index = j;
		}
	}
	return index;
}

double get_2nd_max_coupled(sparse_matrix* A,int i){
	int j,k,J,K;
	double val,val2,p,diag;
	double max = 0;
	int index = -1;
	for (j=0;j<A->Len[i];j++){
		val = A->Values[i][j];
		J = A->Indices[i][j];
		diag = get_diag(A,J);
		if (J!=i && val!=0) for (k=0;k<A->Len[J];k++){
			val2 = A->Values[J][k];
			p = fabs(val*val2/diag);
			K = A->Indices[J][k];
			if (K!=J && K!=i && val2!=0 && p>max){
				max = p;
				index = K;
			}
		}
	}
	return max;
}

void AMG_set_SOR_relaxation_coeff(double alpha){
	SOR_alpha = alpha;
}

void set_connection_thresholds(double negative,double positive){
	eps_minus = negative;
	eps_plus = positive;
}

void print_regular_coarsening(int* Nodes,int* Coarsening,int fine_size,int len){
	int i,j,k,l,m;
	int** Mesh = (int**)malloc(len*sizeof(int*));
	for (i=0;i<len;i++){
		Mesh[i] = (int*)malloc(len*sizeof(int));
		for (j=0;j<len;j++) Mesh[i][j] = FINE;
	}
	m = 0;
	for (l=0;l<fine_size;l++) if (Coarsening[l]!=FINE){
		k = Nodes[m];
		i = k / len;
		j = k % len;
		Mesh[i][j] = COARSE;
		m++;
	}
	FILE* file = fopen("/home/radszuweit/Daten/coarsening","w");
	for (i=0;i<len;i++){
		for (j=0;j<len;j++){
			if (Mesh[i][j]==COARSE) fprintf(file,"* ");
			else fprintf(file,"  ");
		}
		fprintf(file,"\n");
	}
	fclose(file);
}

int* get_strong_connected(sparse_matrix* A,int i,int** Positions,int* Levels){ // first element is the size !
	int j,J; // Achtung nur nodes mit gleichem Freiheitsgrad können connected sein !
	double a;
	int max_index = get_max_coupled(A,i);
	int* Res = (int*)malloc(sizeof(int));
	Positions[i] = (int*)malloc(sizeof(int));
	Res[0] = 0;
	Positions[i][0] = 0;
	if (max_index>=0){
		double min = -eps_minus*fabs(A->Values[i][max_index]);
		double max = eps_plus*fabs(A->Values[i][max_index]);
		for (j=0;j<A->Len[i];j++){
			J = A->Indices[i][j];
			if (J!=i){
				if (Levels[i]==Levels[J]) a = A->Values[i][j]; else a = 0;
				if (a<min || a>max){
					Res[0]++;
					Res = (int*)realloc(Res,(Res[0]+1)*sizeof(int));
					Res[Res[0]] = A->Indices[i][j];
					Positions[i] = (int*)realloc(Positions[i],(Res[0]+1)*sizeof(int));
					Positions[i][Res[0]] = j;
				}
			}
		}
		Positions[i][0] = Res[0];
		return Res;
	}
	else return Res;
}

// neue Version
int* Get_2nd_strong_connected(sparse_matrix* A,int* Label_1st,int i,int* Map,int* Levels){ // first element is the size !
	int j,J,k,K; // Achtung nur nodes mit gleichem Freiheitsgrad können connected sein !
	double a,b,d;
	double max_val = get_2nd_max_coupled(A,i);
	int* Res = (int*)malloc(sizeof(int));
	Res[0] = 0;
	if (max_val>0){
		double min = -eps_minus*max_val;
		double max = eps_plus*max_val;
		for (j=0;j<A->Len[i];j++){
			J = A->Indices[i][j];
			a = A->Values[i][j];
			d = get_diag(A,J);
			if (J!=i && a!=0 && Levels[i]==Levels[J]){
				for (k=0;k<A->Len[J];k++){
					K = A->Indices[J][k];
					b = A->Values[J][k];
					if (Label_1st[K]==COARSE && K!=J && K!=i && a*b/d>0 && Levels[K]==Levels[i]){
						if (-a*b/d<min || a*b/d>max){
							Res[0]++;
							Res = (int*)realloc(Res,(Res[0]+1)*sizeof(int));
							Res[Res[0]] = Map[K];
						}
					}
				}
			}
		}
		return Res;
	}
	else return Res;
}

// alte Version
int* get_2nd_strong_connected(int** Strong_1st,int* Label_1st,int* Map,int i){ // first element is the size !
	int j,k,l,pos;
	int* Res = (int*)malloc(sizeof(int));
	Res[0] = 0;
	for (j=1;j<=Strong_1st[i][0];j++){
		for (k=1;k<=Strong_1st[Strong_1st[i][j]][0];k++){
			l = Strong_1st[Strong_1st[i][j]][k];
			if (l!=i && Label_1st[l]==COARSE){
				pos = in_list(&Res[1],Res[0],Map[l]);
				if (pos<0) {
					Res[0]++;
					Res = (int*)realloc(Res,(Res[0]+1)*sizeof(int));
					Res[Res[0]] = Map[l];
				}
			}
		}
	}
	return Res;
}
 
void get_strong_connection_sets(sparse_matrix* A,int** Matrix,int** Transpose,
 int** Positions,int* Levels){  // first element is the size !
	int i,j,k,m;
	int n = A->size;
	for (i=0;i<n;i++) {
		Transpose[i] = (int*)malloc(sizeof(int));
		Transpose[i][0] = 0;
	}
	for (j=0;j<n;j++){ 
		Matrix[j] = get_strong_connected(A,j,Positions,Levels);
		m = Matrix[j][0];
		for (i=1;i<=m;i++){
			k = Matrix[j][i];
			Transpose[k][0]++;
			Transpose[k] = (int*)realloc(Transpose[k],(Transpose[k][0]+1)*sizeof(int));
			Transpose[k][Transpose[k][0]] = j;
		}
	}
}

void get_2nd_strong_connection_sets(sparse_matrix* A_1st,int** Strong_1st,int* Label_1st,
 int* Map,sparse_matrix* A_2nd,int** Matrix_2nd,int** Transpose_2nd,int* Levels){
	int i,I,j,J,k,K,l,n,m;
	n = A_1st->size;
	for (i=0;i<n;i++){
		if (Label_1st[i]==COARSE){
			I = Map[i];
			Transpose_2nd[I] = (int*)malloc(sizeof(int));
			Transpose_2nd[I][0] = 0;
		}
	}
	for (i=0;i<n;i++){
		if (Label_1st[i]==COARSE){
			I = Map[i];
			//Matrix_2nd[I] = get_2nd_strong_connected(Strong_1st,Label_1st,Map,i);
			Matrix_2nd[I] = Get_2nd_strong_connected(A_1st,Label_1st,i,Map,Levels);
			m = Matrix_2nd[I][0];
			for (k=1;k<=m;k++){
				l = Matrix_2nd[I][k];
				Transpose_2nd[l][0]++;
				Transpose_2nd[l] = (int*)realloc(Transpose_2nd[l],(Transpose_2nd[l][0]+1)*sizeof(int));
				Transpose_2nd[l][Transpose_2nd[l][0]] = I;
			}
			for (j=0;j<A_1st->Len[i];j++){
				J = A_1st->Indices[i][j];
				if (J!=i && A_1st->Values[i][j]!=0) for (k=0;k<A_1st->Len[J];k++){
					K = A_1st->Indices[J][k];
					if (Label_1st[K]==COARSE && K!=J && K!=i && A_1st->Values[J][k]!=0) insert_sparse(A_2nd,1.,I,Map[K]);
				}
			}
		}
	}
}

sparse_matrix* AMG_Strong_reduced(sparse_matrix* A,int** Strong,int** Positions){
	int i,j,k;
	int n = A->size;
	sparse_matrix* R = sparse_zero(n);
	for (i=0;i<n;i++){
		for (j=1;j<=Strong[i][0];j++){
			k = Strong[i][j];
			insert_sparse(R,A->Values[i][Positions[i][j]],i,k);
		}
	}
	delete_zeros(R);
	return R;
}

sparse_matrix* get_AMG_transposed(sparse_matrix* A,int tra_size){
	int i,j,k;
	double a;
	sparse_matrix* T = sparse_zero(tra_size);
	for (i=0;i<A->size;i++){
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			a = A->Values[i][j];
			if (a!=0) insert_sparse(T,a,k,i);
		}
	}
	delete_zeros(T);
	return T;
}

int get_AMG_index_map(int* Labels,int* Map,int* Inverse,int size,int with_isolated){  // Achtung: Map muss Größe fine_size, Inverse coarse_size haben
	int i;
	int k = 0;
	for (i=0;i<size;i++){
		if (Labels[i]!=FINE){
			if (with_isolated || Labels[i]==COARSE){
				if (Map!=NULL) Map[i] = k;
				if (Inverse!=NULL) Inverse[k] = i;
				k++;
			}
			else Map[i] = -1;
		}
		else Map[i] = -1;
	}
	return k;
}

int AMG_rb_comp_double(const void* A,const void* B){
	double* a = (double*)A;
	double* b = (double*)B;
	if ((*a)>(*b)) return 1; else return 0;
}

void AMG_rb_print_key(const void* A){
	double* a = (double*)A;
	printf("%f  |",*a);
}

void AMG_rb_print_info(void* I){
	int* i = (int*)I;
	printf("%d\n",*i);
}

void AMG_rb_free_key(void* A){
}

void AMG_rb_free_info(void* I){
	int* i = (int*)I;
	free(i);
}

double AMG_get_lambda(int* Label,int** Transpose,int i,int size){
	int j,J;
	int undec = 0;
	int fine = 0;
	for (j=1;j<=Transpose[i][0];j++){
		J = Transpose[i][j];
		if (Label[J]==UNDECIDED) undec++;
		if (Label[J]==FINE) fine++;
	}
	double res = (double)i/size+undec+2*fine;
	return res;
}

void AMG_coarsener(sparse_matrix* A,int* Label,double* Importance,int** Strong_connections,
 int** Transpose,rb_red_blk_node** Node_list,int* Levels,int size,int seed){		// Label alle auf UNDECIDED
	int i,I,j,J,next_coarse;											// Node_list alle auf NULL
	int* Ind;
	double lambda;
	/*int done = 0;
	int level_size = 0;
	for (i=0;i<size;i++) if (Levels[i]==level){
		level_size++;
		if (Label[i]==ISOLATED) done++;
	}*/
	rb_red_blk_node* Doomed;
	int level = Levels[seed];
	rb_red_blk_tree* Importance_tree = RBTreeCreate(&AMG_rb_comp_double,&AMG_rb_free_key,
	 &AMG_rb_free_info,&AMG_rb_print_key,&AMG_rb_print_info);
	next_coarse = seed;
	do{
		Label[next_coarse] = COARSE;
		//done++;
		//printf("%d COARSE\n",next_coarse);							//debug !
		for (i=0;i<A->Len[next_coarse];i++){
			I = A->Indices[next_coarse][i];
			if (Label[I]==UNDECIDED && A->Values[next_coarse][i]!=0. && Node_list[I]==NULL){
				Ind = (int*)malloc(sizeof(int));
				*Ind = I;
				Node_list[I] = RBTreeInsert(Importance_tree,&Importance[I],Ind);
			}
		}
		for (i=1;i<=Transpose[next_coarse][0];i++){
			I = Transpose[next_coarse][i];
			if (Node_list[I]!=NULL){
				Label[I] = FINE;
				//printf("%d -> %d FINE\n",next_coarse,I);			//debug !
				RBDelete(Importance_tree,Node_list[I]);
				Node_list[I] = NULL;
				//if (Levels[I]==level) done++;
			}
		}
		for (i=0;i<A->Len[next_coarse];i++){
			I = A->Indices[next_coarse][i];
			if (A->Values[next_coarse][i]!=0. && Node_list[I]!=NULL){
				lambda = AMG_get_lambda(Label,Transpose,I,size);
				if (lambda!=Importance[I]){
					RBDelete(Importance_tree,Node_list[I]);
					Ind = (int*)malloc(sizeof(int));
					*Ind = I;
					Importance[I] = lambda;
					Node_list[I] = RBTreeInsert(Importance_tree,&Importance[I],Ind);
				}
			}
		}
		for (i=0;i<A->Len[next_coarse];i++){
			I = A->Indices[next_coarse][i];
			if (I!=next_coarse){
				for (j=0;j<A->Len[I];j++){
					J = A->Indices[I][j];
					if (Label[J]==UNDECIDED && A->Values[I][j]!=0. && Node_list[J]==NULL){
						Ind = (int*)malloc(sizeof(int));
						*Ind = J;
						Node_list[J] = RBTreeInsert(Importance_tree,&Importance[J],Ind);
					}
				}
				for (j=1;j<=Strong_connections[I][0];j++){
					J = Strong_connections[I][j];
					if (Label[J]==UNDECIDED){
						lambda = AMG_get_lambda(Label,Transpose,J,size);
						if (lambda!=Importance[J]){
							RBDelete(Importance_tree,Node_list[J]);
							Ind = (int*)malloc(sizeof(int));
							*Ind = J;
							Importance[J] = lambda;
							Node_list[J] = RBTreeInsert(Importance_tree,&Importance[J],Ind);
						}
					}
				}
			}
		}
		Doomed = RBLargest(Importance_tree);
		if (Doomed!=NULL) {
			next_coarse = *(int*)Doomed->info;
			RBDelete(Importance_tree,Doomed);
			Node_list[next_coarse] = NULL;
		}
		else{
			next_coarse = -1;
			for (i=0;i<size;i++) if (Label[i]==UNDECIDED && Levels[i]==level){
					next_coarse = i;
					break;
			}
		}
	}while(next_coarse>=0);
	RBTreeDestroy(Importance_tree);
	/*if (done!=level_size){
		printf("coarsening error -> abort\n");
		exit(0);
	}*/
 }
 
 sparse_matrix* get_restricted_matrix(sparse_matrix* A,sparse_matrix* R,sparse_matrix* P){
	sparse_matrix* C = sparse_product(A,P);
	sparse_matrix* Res = sparse_product(R,C);
	free_sparse(C);
	if (!square_matrix_clean(Res)){
		printf("erroneous coarse_matrix -> abort\n");
		exit(0);
	}
	return Res;
}
 
 void AMG_pure_restriction(int* Label,int n,sparse_matrix* R,sparse_matrix* RT){
	int i,k;
	k = 0;
	for (i=0;i<n;i++){
		if (Label[i]==COARSE){
			if (R!=NULL) insert_sparse(R,1.,k,i);
			if (RT!=NULL) insert_sparse(RT,1.,i,k);
			k++;
		}
	}
}

int AMG_is_isolated(sparse_matrix* A,int i){
	int j;
	int n = 0;
	for (j=0;j<A->Len[i];j++) if (A->Values[i][j]!=0) n++;
	return (n==1?1:0);
}


int standard_coarsening(sparse_matrix* A,int* Label,int** Strong_connections,int** Positions,
 int** Transpose,int* Levels,int mode){
	int i,j,seed;
	int n = A->size;
	int iso_number = 0;
	double* Importance = (double*)malloc(n*sizeof(double));
	rb_red_blk_node** Node_list = (rb_red_blk_node**)malloc(n*sizeof(rb_red_blk_node*));
	get_strong_connection_sets(A,Strong_connections,Transpose,Positions,Levels);
	int bad = 0;
	for (i=0;i<n;i++){
		if (Strong_connections[i][0]==0){
			if (AMG_is_isolated(A,i)){
				Label[i] = ISOLATED; 
				iso_number++;
			}
			else{
				bad++;
				Label[i] = COARSE;
			}
		}
		else Label[i] = UNDECIDED;
		Importance[i] = 0.;
		Node_list[i] = NULL;
	}
	for (i=0;i<AMG_deg_of_freedom;i++){
		seed = -1;
		for (j=0;j<n;j++) if (Label[j]==UNDECIDED && Levels[j]==i){
			seed = j;
			break;
		}
		if (seed>=0) AMG_coarsener(A,Label,Importance,Strong_connections,Transpose,Node_list,Levels,n,seed);
	}
	int coarse_number = 0;
	for (i=0;i<n;i++){
		if (Label[i]==UNDECIDED) Label[i] = FINE;
		if (Label[i]==COARSE) coarse_number++;
	}
	if (mode==STAND_ALONE) for (i=0;i<n;i++) if (Label[i]==ISOLATED) coarse_number++;
	free(Importance);
	free(Node_list);
	if (AMG_info) printf("badly connected nodes: %f\n",(double)bad/n);
	return coarse_number;
}

int aggressive_coarsening(sparse_matrix* A,int* Label,int** Strong_connections,int** Positions,
 int** Transpose,int* Levels){
	int i,I,pre_size,seed;
	int n = A->size;
	int coarse_size = 0;
	int* Map = (int*)malloc(n*sizeof(int));
	int* Pre_Label = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++) Pre_Label[i] = UNDECIDED;
	pre_size = standard_coarsening(A,Pre_Label,Strong_connections,Positions,Transpose,Levels,PREMODE);
	for (i=0;i<n;i++) Label[i] = Pre_Label[i];
	int* Inv_Map = (int*)malloc(pre_size*sizeof(int));
	get_AMG_index_map(Pre_Label,Map,Inv_Map,n,WITHOUT_IOSOLATED);
	sparse_matrix* Pre_A = sparse_zero(pre_size);
	int* Label_2nd = (int*)malloc(pre_size*sizeof(int));
	int** Matrix_2nd = (int**)malloc(pre_size*sizeof(int*));
	int** Transpose_2nd = (int**)malloc(pre_size*sizeof(int*));
	double* Importance_2nd = (double*)malloc(pre_size*sizeof(double));
	rb_red_blk_node** Node_list = (rb_red_blk_node**)malloc(pre_size*sizeof(rb_red_blk_node*));
	for (i=0;i<pre_size;i++){
		Node_list[i] = NULL;
		Label_2nd[i] = UNDECIDED;
	}
	get_2nd_strong_connection_sets(A,Strong_connections,Pre_Label,Map,Pre_A,
	 Matrix_2nd,Transpose_2nd,Levels);
	for (i=0;i<AMG_deg_of_freedom;i++){
		seed = -1;
		for (I=0;I<pre_size;I++) if (Levels[Inv_Map[I]]==i){
			seed = I;
			break;
		}
		AMG_coarsener(Pre_A,Label_2nd,Importance_2nd,Matrix_2nd,Transpose_2nd,Node_list,Levels,
		 pre_size,seed);
	}
	for (i=0;i<n;i++){
		if (Label[i]==COARSE){
			I = Map[i];
			Label[i] = Label_2nd[I];
			if (Label[i]==COARSE) coarse_size++;
		}
		else if (Label[i]==ISOLATED) coarse_size++;
	}
	
	free_sparse(Pre_A);
	for (i=0;i<pre_size;i++){
		free(Matrix_2nd[i]);
		free(Transpose_2nd[i]);
	}
	free(Matrix_2nd);
	free(Transpose_2nd);
	free(Importance_2nd);
	free(Label_2nd);
	free(Pre_Label);
	free(Node_list);
	free(Map);
	free(Inv_Map);
	AMG_coarsening = AGGRESSIVE;
	return coarse_size;
}

int AMG_average_strong_con(int** Transpose,int* Label,int list_size){
	int i;
	int size = 0;
	int sum = 0;
	for (i=0;i<list_size;i++) if (Label[i]!=ISOLATED){
		sum += Transpose[i][0];
		size++;
	}
	return sum/size;
}

void test_coarse(sparse_matrix* A,int* Label){
	int i,j,k;
	int n = A->size;
	for (i=0;i<n;i++){
		for (j=0;j<A->Len[i];j++){
			k = A->Indices[i][j];
			if (k!=i && Label[i]==COARSE && Label[k]==COARSE) printf("Coarse-Nachbarn %d-%d\n",i,k);
		}
	}
}

int get_AMG_coarse_size(int* Labels,int size){
	int i;
	int res = 0;
	for (i=0;i<size;i++) if (Labels[i]==COARSE) res++;
	return res;
}

void set_AMG_smoothing_iterations(int fine_iter,int coarse_iter){
	AMG_smoothing_iterations = fine_iter;
	AMG_aggr_smoothing_iterations = coarse_iter;
}

int get_AMG_col_number(sparse_matrix* A,int i,int j){
	int* Flag = (int*)malloc(sizeof(int));
	int res = get_position(A->Indices[i],A->Len[i],j,Flag);
	if (*Flag>0) return res;
	else return -1;
}

void AMG_print_info(int s){
	AMG_info = s;
}

int get_direct_matrix_elements(sparse_matrix* A,sparse_matrix* B,int* Label,int i,
int** Indices,double** Values,int** Status,int** Strong,int** Positions){
	int j,pos;
	int n = A->Len[i];
	*Status = (int*)malloc(n*sizeof(int));
	for (j=0;j<n;j++) (*Status)[j] = WEAK;
	*Indices = clone_list(A->Indices[i],n);
	*Values = clone_vector(A->Values[i],n);
	for (j=1;j<=Strong[i][0];j++){
		pos = Positions[i][j];
		if (Label[Strong[i][j]]==COARSE) (*Status)[pos] = COARSE;
		else (*Status)[pos] = FINE;
	}
	return n;
}

int get_standard_matrix_elements(sparse_matrix* A,sparse_matrix* B,int* Label,int i,
int** Indices,double** Values,int** Status,int** Strong,int** Positions){
	int m,j,k,h,g,pos,size;
	double d,q;
	int n = A->Len[i];
	*Status = (int*)malloc(n*sizeof(int));
	for (j=0;j<n;j++) (*Status)[j] = WEAK;
	*Indices = clone_list(A->Indices[i],n);
	*Values = clone_vector(A->Values[i],n);
	for (j=1;j<=Strong[i][0];j++){
		pos = Positions[i][j];
		if (Label[Strong[i][j]]==COARSE) (*Status)[pos] = COARSE;
		else (*Status)[pos] = FINE;
	}
	size = n;
	for (h=0;h<n;h++){
		j = (*Indices)[h];
		m = A->Len[j];
		if (Label[j]==FINE && j!=i){
			d = get_diag(A,j);
			if (d==0){
				printf("matrix has zero diagonal at index %d  AMG not possible -> abort\n",h);
				exit(0);
			}
			(*Values)[h] = 0;
			q = A->Values[i][h]/d;
			if (B!=NULL) insert_sparse(B,q,i,j);
			for (g=0;g<m;g++){
				k = A->Indices[j][g];
				if (k!=j){
					pos = in_list(*Indices,size,k);
					if (pos<0){
						size++;
						*Indices = (int*)realloc(*Indices,size*sizeof(int));
						*Values = (double*)realloc(*Values,size*sizeof(double));
						*Status = (int*)realloc(*Status,size*sizeof(int));
						(*Indices)[size-1] = k;
						(*Values)[size-1] = -(A->Values[j][g])*q;
						(*Status)[size-1] = Label[k];
					}
					else{
						(*Values)[pos] -= (A->Values[j][g])*q;
					}
				}
			}
		}
	}
	return size;
}

void set_prolongation_row(sparse_matrix* A,sparse_matrix* P,sparse_matrix* B,int i,
int* Label,int** Strong,int** Positions,int* Map){
	int j,size;
	double alpha,beta,p,aik;
	double diag_val = 0;
	double coarse_minus = 0;
	double coarse_plus = 0;
	double sum_minus = 0;
	double sum_plus = 0;
	int** Indices = (int**)malloc(sizeof(int*));
	int** Status = (int**)malloc(sizeof(int*));
	double** Values = (double**)malloc(sizeof(double*));
	size = (*F_Prolong)(A,B,Label,i,Indices,Values,Status,Strong,Positions);
	for (j=0;j<size;j++){
		aik = (*Values)[j];
		if ((*Indices)[j]==i){
			diag_val = aik;
			continue;
		}
		if (aik<0) sum_minus += aik;
		else sum_plus += aik;
		if ((*Status)[j]==COARSE){
			if (aik<0) coarse_minus += aik;
			else coarse_plus += aik;
		}
	}
	if (diag_val==0){
		printf("diagonal element is zero at index %d -> abort\n",i);
		exit(0);
	}
	if (coarse_minus==0){
		printf("interpolation of var %d not possible -> abort\n",i);
		exit(0);
	}
	else alpha = sum_minus/coarse_minus;
	if (coarse_plus==0){
		diag_val += sum_plus;
		beta = 0;
	}
	else beta = sum_plus/coarse_plus;
	int interpolation_number = 0;
	for (j=0;j<size;j++){
		if ((*Status)[j]==COARSE){
			interpolation_number++;
			aik = (*Values)[j];
			if (aik<0) p = -alpha*aik/diag_val;
			else p = -beta*aik/diag_val;
			insert_sparse(P,p,i,Map[(*Indices)[j]]);
		}
	}
	if (interpolation_number==0) printf("Warning: index %d has no interpolation nodes\n",i);
	if (B!=NULL) insert_sparse(B,1./diag_val,i,i);
	
	free(*Indices);
	free(*Values);
	free(*Status);
	free(Indices);
	free(Values);
	free(Status);
}

void AMG_init_finest_tupels(int** Tupels,int* Levels,int* Nodes,int size){
	int i,j,k;
	int mesh_size = size / AMG_deg_of_freedom;
	for (i=0;i<size;i++){
		if (i<mesh_size){
			Tupels[i] = (int*)malloc((AMG_deg_of_freedom+1)*sizeof(int));
			Tupels[i][0] = AMG_deg_of_freedom;
			for (j=1;j<=AMG_deg_of_freedom;j++){
				k = (j-1)*mesh_size+i;
				Tupels[i][j] = k;
				Levels[k] = j-1;
				Nodes[k] = i;
			}
		}
		else{
			Tupels[i] = (int*)malloc(sizeof(int));
			Tupels[i][0] = 0;
		}
	}
}

void AMG_set_tupels(int* Label,int* New_levels,int* Old_levels,int* New_nodes,int* Old_nodes,
 int** New_tupels,int** Old_tupels,int fine_size){
	int i;
	int* Map = (int*)malloc(fine_size*sizeof(int));
	int coarse_size = get_AMG_index_map(Label,Map,NULL,fine_size,WITH_ISOLATED);
	for (i=0;i<coarse_size;i++){
		New_tupels[i] = (int*)malloc(sizeof(int));
		New_tupels[i][0] = 0;
	}
	for (i=0;i<fine_size;i++){
		if (Map[i]>=0){
			New_levels[Map[i]] = Old_levels[i];
			New_nodes[Map[i]] = Old_nodes[i];
		}
		/*L = -1;
		for (j=1;j<=Old_tupels[i][0];j++){
			k = Old_tupels[i][j];
			if (Label[k]!=FINE){
				if (L<0) L = Map[k];
				n = New_tupels[L][0]+1;
				New_tupels[L]= (int*)realloc(New_tupels[L],(n+1)*sizeof(int));
				New_tupels[L][n] = Map[k];
				New_tupels[L][0] = n;
			}
		}*/
	}
	free(Map);
}

void AMG_standard_interpolation(sparse_matrix* A,sparse_matrix* P,int* Label,
 int** Strong,int** Positions,int** Tupels,int* Levels,int* Nodes,int* Map,int iter){
	int i,I,j,k,l,J,L,m,size,t;
	int zero_case = 1;
	int* Indices;
	int* Ind;
	double** alpha;
	double** beta;
	double** G;
	double** Inv_G;
	double* s_m;
	double* s_p;
	double* c_m;
	double* c_p;
	double a,q;
	int n = A->size;
	double* D = zero_vector(n);
	double* Values;
	sparse_matrix* Q = clone(A);
	for (i=0;i<n;i++){
		D[i] = get_diag(A,i);
		if (D[i]==0){
			printf("diagonal zero at index %d standard interpolation not possible -> abort\n",i);
			exit(0);
		}
		for (j=0;j<Q->Len[i];j++) Q->Values[i][j] /= D[i];
	}
	for (i=0;i<n;i++){
		if (Label[i]==FINE) for (k=0;k<iter;k++){
			size = Q->Len[i];
			Indices = clone_list(Q->Indices[i],size);
			Values = clone_vector(Q->Values[i],size);
			for (j=0;j<size;j++){
				J = Indices[j];
				q = Values[j];
				if (J!=i && Label[J]==FINE && q!=0){
					remove_element(Q,i,J);
					for (l=0;l<A->Len[J];l++){
						L = A->Indices[J][l];
						a = A->Values[J][l]/D[J];
						if (L!=J && a!=0) insert_sparse(Q,-a*q,i,L);
					}
				}
			}
			free(Indices);
			free(Values);
		}
		else{
			reset_row(Q,i);
			insert_sparse(Q,1.,i,i);
		}
	}
	for (i=0;i<n;i++) if (Tupels[i][0]>0){
		t = Tupels[i][0];
		Ind = (int*)malloc(AMG_deg_of_freedom*sizeof(int));
		for (m=0;m<AMG_deg_of_freedom;m++) Ind[m] = -1;
		for (m=0;m<t;m++){
			l = Levels[Tupels[i][m+1]];
			Ind[l] = m;
		}
		G = create_2D_array(t,t);
		alpha = create_2D_array(t,AMG_deg_of_freedom);
		beta = create_2D_array(t,AMG_deg_of_freedom);
		for (k=0;k<t;k++){
			I = Tupels[i][k+1];
			if (Label[I]==FINE){
				s_p = zero_vector(AMG_deg_of_freedom);
				c_p = zero_vector(AMG_deg_of_freedom);
				s_m = zero_vector(AMG_deg_of_freedom);
				c_m = zero_vector(AMG_deg_of_freedom);
				for (j=0;j<Q->Len[I];j++){	// noch nicht fertig:
					J = Q->Indices[I][j]; // hier nachprüfen ob J strong connected ist !
					q = Q->Values[I][j];
					l = Levels[J];
					if (Nodes[J]!=Nodes[I]){
						if (q<0){
							s_m[l] += q;
							if (Label[J]==COARSE) c_m[l] += q;
						}
						else{
							s_p[l] += q;
							if (Label[J]==COARSE) c_p[l] += q;
						}
					}
					else{
						if (Ind[l]>=0) G[k][Ind[l]] = q;
						//else printf("wrong index at %d\n",i);
					}
				}
				for (l=0;l<AMG_deg_of_freedom;l++){
					if (c_m[l]!=0) alpha[k][l] = -s_m[l]/c_m[l];
					else{
						if (Ind[l]>=0) G[k][Ind[l]] +=s_m[l];
						//else printf("wrong index at %d\n",i);
					}
					if (c_p[l]!=0) beta[k][l] = -s_p[l]/c_p[l]; 
					else{
						if (Ind[l]>=0) G[k][Ind[l]] +=s_p[l];
						//else printf("wrong index at %d\n",i);
					}
				}
				free(s_m);
				free(s_p);
				free(c_m);
				free(c_p);
			}
			else{
				G[k][k] = 1.;
				alpha[k][k] = 1.;
			}
		}
		for (k=0;k<t;k++){
			for (l=0;l<AMG_deg_of_freedom;l++) if (alpha[k][l]!=0 || beta[k][l]!=0) zero_case = 0;
		}
		if (zero_case){
			printf("no coarse interpolation nodes present at mesh-index %d -> abort\n",i);
			exit(0);
		}
		Inv_G = get_inverse(G,t);
		for (k=0;k<t;k++){
			I = Tupels[i][k+1];
			if (Label[I]==FINE){
				for (l=0;l<t;l++){
					J = Tupels[i][l+1];
					for  (j=0;j<Q->Len[J];j++){
						L = Q->Indices[J][j];			//hier auch !
						q = Q->Values[J][j];
						m = Levels[L];
						if (Nodes[L]!=Nodes[I] && Label[L]==COARSE){
							if (q<0 && alpha[l][m]!=0) insert_sparse(P,Inv_G[k][l]*alpha[l][m]*q,I,Map[L]);
							if (q>0 && beta[l][m]!=0) insert_sparse(P,Inv_G[k][l]*beta[l][m]*q,I,Map[L]);
						}
					}
				}
			}
			else insert_sparse(P,1.,I,Map[I]);
		}
		for (k=0;k<t;k++){
			free(alpha[k]);
			free(beta[k]);
			free(G[k]);
			free(Inv_G[k]);
		}
		free(alpha);
		free(beta);
		free(G);
		free(Inv_G);
		free(Ind);
	}
	free(D);
	free_sparse(Q);
	delete_zeros(P);
}

void AMG_simple_standard_interpolation(sparse_matrix* A,sparse_matrix* P,int* Label,
 int** Strong,int** Positions,int** Tupels,int* Levels,int* Nodes,int* Map,int iter){
	int i,j,k,l,J,L,size;
	int* Indices;
	double a,q,s_m,s_p,c_m,c_p,alpha,beta;
	int n = A->size;
	double* D = zero_vector(n);
	double* Values;
	sparse_matrix* Q = clone(A);// alternative: AMG_Strong_reduced(A,Strong,Positions);
	for (i=0;i<n;i++){
		D[i] = get_diag(A,i);
		if (D[i]==0){
			printf("diagonal zero at index %d standard interpolation not possible -> abort\n",i);
			exit(0);
		}
		for (j=0;j<Q->Len[i];j++) Q->Values[i][j] /= D[i];
	}
	for (i=0;i<n;i++){
		if (Label[i]==FINE) for (k=0;k<iter;k++){
			size = Q->Len[i];
			Indices = clone_list(Q->Indices[i],size);
			Values = clone_vector(Q->Values[i],size);
			for (j=0;j<size;j++){
				J = Indices[j];
				q = Values[j];
				if (J!=i && Label[J]==FINE && q!=0){
					remove_element(Q,i,J);
					for (l=0;l<A->Len[J];l++){
						L = A->Indices[J][l];
						a = A->Values[J][l]/D[J];
						if (L!=J && a!=0) insert_sparse(Q,-a*q,i,L);
					}
				}
			}
			free(Indices);
			free(Values);
		}
	}
	for (i=0;i<n;i++){
		if (Label[i]==FINE){
			j = 0;
			while (j<Q->Len[i]){
				if (Levels[Q->Indices[i][j]]!=Levels[i]){
					remove_element_at(Q,i,j);
				}
				else j++;
			}
			s_p = 0;c_p = 0;
			s_m = 0;c_m = 0;
			for (j=0;j<Q->Len[i];j++){	// noch nicht fertig:
				J = Q->Indices[i][j]; // hier nachprüfen ob J strong connected ist !
				q = Q->Values[i][j];
				if (J!=i){
					if (q<0){
						s_m += q;
						if (Label[J]==COARSE) c_m += q;
					}
					else{
						s_p += q;
						if (Label[J]==COARSE) c_p += q;
					}
				}
				else D[i] = q;
			}
			alpha = 0;
			beta = 0;
			if (c_m!=0) alpha = -s_m/c_m; else D[i] += s_m;
			if (c_p!=0) beta = -s_p/c_p; else D[i] += s_p;
			if (alpha==0 && beta==0){
				printf("no coarse interpolation nodes present at index %d -> abort\n",i);
				break;
			}
			else{
				alpha /= D[i];
				beta /= D[i];
				for (j=0;j<Q->Len[i];j++){
					J = Q->Indices[i][j];			//hier auch !
					q = Q->Values[i][j];
					if (J!=i && Label[J]==COARSE){
						if (q<0 && alpha!=0) insert_sparse(P,alpha*q,i,Map[J]);
						if (q>0 && beta!=0) insert_sparse(P,beta*q,i,Map[J]);
					}
				}
			}
		}
		else insert_sparse(P,1.,i,Map[i]);
	}
	free(D);
	free_sparse(Q);
	delete_zeros(P);
}

int AMG_get_next(int* Label,int size,int** Strong_transpose,int** Next,int* Prev,int prev_size){
	int i,I,j,J;
	int next_size = 0;
	int* Taken = clone_list(Label,size);
	for (i=0;i<prev_size;i++){
		I = Prev[i];
		for (j=1;j<=Strong_transpose[I][0];j++){
			J = Strong_transpose[I][j];
			if (Taken[J]==FINE){
				next_size++;
				*Next = (int*)realloc(*Next,next_size*sizeof(int));
				(*Next)[next_size-1] = J;
				Taken[J] = COARSE;
			}
		}
	}
	if (next_size==0) for (i=0;i<size;i++) if (Taken[i]==FINE){
		next_size++;
		*Next = (int*)realloc(*Next,next_size*sizeof(int));
		(*Next)[next_size-1] = i;
		Taken[i] = COARSE;
		break;
	}
	free(Taken);
	return next_size;
}

void AMG_multi_pass_interpolation(int* Coarsening,int coarse_size,sparse_matrix* A,
 sparse_matrix* P,sparse_matrix* B,int** Strong,int** Transpose,int** Positions,int* Map,int* Levels){
	int i,I,j,J,k,K;
	int next_size = 0;
	int prev_size = coarse_size;
	int n = P->size;
	double sum_minus,sum_plus,val,coarse_minus,coarse_plus,diag,alpha,beta,p;
	int** Next = (int**)malloc(sizeof(int*));
	int** Prev = (int**)malloc(sizeof(int*));
	*Next = NULL;
	*Prev = (int*)malloc(coarse_size*sizeof(int));
	int* Label = clone_list(Coarsening,n);
	j = 0;
	for (i=0;i<n;i++){
		if (Label[i]!=FINE){
			(*Prev)[j] = i;
			insert_sparse(P,1.,i,i);
			j++;
		}
	}
	next_size = AMG_get_next(Label,n,Transpose,Next,*Prev,prev_size);
	while(next_size>0){
		for (i=0;i<next_size;i++){
			I = (*Next)[i];
			if (I==10542){
				printf("breakpoint\n");
			}
			sum_minus = 0;
			sum_plus = 0;
			coarse_minus = 0;
			coarse_plus = 0;
			diag = 0;
			for (j=0;j<A->Len[I];j++){
				J = A->Indices[I][j];
				if (Levels[J]==Levels[I]){
					val = A->Values[I][j];
					if (J==I) diag = val;
					else{
						if (val<0) sum_minus += val; else sum_plus += val;
					}
				}
			}
			for (j=1;j<=Strong[I][0];j++){
				J = Strong[I][j];
				if (Label[J]!=FINE){
					val = A->Values[I][Positions[I][j]];
					if (val<0) coarse_minus += val; else coarse_plus += val;
				}
			}
			if (coarse_minus!=0 || coarse_plus!=0){
				if (coarse_minus!=0) alpha = sum_minus/coarse_minus;
				else{
					diag += sum_minus;
					alpha = 0;
				}
				if (coarse_plus!=0) beta = sum_plus/coarse_plus;
				else{
					diag += sum_plus;
					beta = 0;
				}
				if (diag==0){
					printf("Error in multi-pass interpolation: zero diagonal at index %d ->abort!\n",I);
					exit(0);
				}
				if (B!=NULL) insert_sparse(B,1./diag,I,I);
				for (j=1;j<=Strong[I][0];j++){
					J = Strong[I][j];
					if (Label[J]!=FINE && Levels[J]==Levels[I]){
						val = A->Values[I][Positions[I][j]];
						if (val<0) val *= -alpha/diag;
						else val *= -beta/diag;
						for (k=0;k<P->Len[J];k++){
							K = P->Indices[J][k];
							p = P->Values[J][k];
							if (p!=0) insert_sparse(P,p*val,I,K);  
						}
						if (B!=NULL) for (k=0;k<B->Len[J];k++){
							K = B->Indices[J][k];
							p = B->Values[J][k];
							if (p!=0) insert_sparse(B,p*val,I,K);
						}
					}
				}
			}
			else{
				printf("Error in multi-pass interpolation: no coarse points near index %d ->abort!\n",I);
				exit(0); 
			}
		}
		for (i=0;i<next_size;i++) Label[(*Next)[i]] = COARSE;
		free(*Prev);
		*Prev = *Next;
		*Next = NULL;
		prev_size = next_size;
		next_size = AMG_get_next(Label,n,Transpose,Next,*Prev,prev_size);
	};
	for (i=0;i<n;i++){
		for (j=0;j<P->Len[i];j++){
			J = P->Indices[i][j];
			K = Map[J];
			if (P->Values[i][j]!=0){
				if (K<0 || K>=coarse_size){
					printf("Error in multi-pass interpolation: erroneous interpolation matrix at index %d|%d ->abort!\n",i,J);
					exit(0);
				}
				P->Indices[i][j] = K;
			}
		}
	}
	delete_zeros(P);
	if (B!=NULL) delete_zeros(B);
	if (*Next!=NULL) free(*Next);
	if (*Prev!=NULL) free(*Prev);
	free(Prev);
	free(Next);
	free(Label);
}

sparse_matrix* Jacobi_post_smoothing(int* Coarsening,sparse_matrix* A,sparse_matrix* P,
 sparse_matrix* B,int** Strong,int** Positions){
	int i,j,J,k,pos;
	double a,b,sum,alpha_p,alpha_m,val,diag;
	int n = P->size;
	sparse_matrix* S = sparse_zero(n);
	for (i=0;i<n;i++) if (Coarsening[i]==FINE){
		alpha_m = 0;
		alpha_p = 0;
		diag = 0;
		a = 0;
		for (j=0;j<A->Len[i];j++) if (A->Indices[i][j]!=i){
			val = A->Values[i][j];
			if (val<0) alpha_m += val; else alpha_p += val;
		}
		else diag = A->Values[i][j];
		sum = 0;
		for (j=1;j<=Strong[i][0];j++) sum += A->Values[i][Positions[i][j]];
		a = alpha_m/((diag+alpha_p)*sum);
		for (j=1;j<=Strong[i][0];j++){
			pos = Positions[i][j];
			J = A->Indices[i][pos];
			a *= -A->Values[i][pos];
			if (a!=0) for (k=0;k<P->Len[J];k++){
				b = P->Values[J][k];
				if (b!=0) insert_sparse(S,a*b,i,P->Indices[J][k]);
			}
		}
		insert_sparse(B,1./(diag+alpha_p),i,i);
	}
	else{
		val = P->Values[i][0];
		if (val!=0)insert_sparse(S,val,i,P->Indices[i][0]);
		else if (Coarsening[i]==COARSE){
			printf("Fehler bei Prolong smoothing: diag zero\n");
			print_sparse(P);
			exit(0);
		}
	}
	return S;
}

void set_prolongation_map(int* Labels,sparse_matrix* A,sparse_matrix* P,int** Strong,
 int** Positions,int** Transpose,int** Tupels,int* Levels,int* Nodes,int coarsening_type){
												// für b Nullvektor reingeben
	int n = A->size;
	int* Map = (int*)malloc(n*sizeof(int));
	int coarse_size = get_AMG_index_map(Labels,Map,NULL,n,WITH_ISOLATED);
	if (coarsening_type==STANDARD){
		AMG_simple_standard_interpolation(A,P,Labels,Strong,Positions,Tupels,Levels,
		 Nodes,Map,1);
		sparse_approximate(P,AMG_tol);
	}
	else{ 
		AMG_multi_pass_interpolation(Labels,coarse_size,A,P,NULL,Strong,Transpose,
		 Positions,Map,Levels);
		sparse_approximate(P,AMG_tol);
	}
	free(Map);
}

void AMG_settings(int deg_of_freedom,int smoothing_iterations,int aggr_smoothing_iterations,
 int coarsening,int AMG_cycle_structure){
	AMG_deg_of_freedom = deg_of_freedom;
	AMG_smoothing_iterations = smoothing_iterations;
	AMG_aggr_smoothing_iterations = aggr_smoothing_iterations;
	AMG_coarsening = coarsening;
	switch(AMG_cycle_structure){
		case DIRECT: F_Prolong = &get_direct_matrix_elements;break;
		case STANDARD: F_Prolong = &get_standard_matrix_elements;break;
		default: F_Prolong = &get_direct_matrix_elements;
	}
}

void AMG_fix_isolated(sparse_matrix* A,double* F,double* Sol,int* Label){
	int i,j;
	double a;
	for (i=0;i<A->size;i++) if (Label[i]==ISOLATED){
		a = 0;
		for (j=0;j<A->Len[i];j++) if (A->Values[i][j]!=0){
			a = A->Values[i][j];
			break;
		}
		if (a==0){
			if (F[i]!=0) printf("System has no solution (index %d) -> abort\n",i);
			else printf("System has no unique solution: trivial row at index %d -> abort\n",i);
			exit(0);
		}
		else{
			Sol[i] = F[i]/a;
		}
	}
}

double* test_direct(sparse_matrix* A,int* Label,double* F){
	int i;
	double d;
	int n = A->size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++) if (Label[i]==FINE){
		d = get_diag(A,i);
		Res[i] = F[i]/d;
	}
	return Res;
}

int* get_GS_Sequence(int** Con_sets,int n){
	int i;
	int* Indices = (int*)malloc(n*sizeof(int));
	double* Weights = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		Indices[i] = i;
		Weights[i] = (double)Con_sets[i][0];
	}
	//double_sort(Indices,Weights,n);
	return Indices;
}

void AMG_smoother(sparse_matrix* A,double* F,double* Sol,int* Label,double alpha,int fix,int iter){
	int i,j,k,ind,diag;							//SOR-Verfahren
	double sum;
	int n = A->size;
	for (k=0;k<iter;k++){
		//#pragma omp parallel shared(A) private(i,sum)
		for (i=0;i<n;i++) if (Label[i]!=fix){
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

void AMG_LU_solver(sparse_matrix* A,double* F,double* Sol){   // simple LU-factorization
	int i;
	int size = A->size;
	double** M = convert_sparse_to_array(A,size);
	double** L = (double**)malloc(size*sizeof(double*));
	double** U = (double**)malloc(size*sizeof(double*));
	LU_factorzation(M,L,U,size);
	double* G = Lower_triangular_invert(L,F,size);
	double* H = Upper_triangular_invert(U,G,size);
	for (i=0;i<size;i++){
		free(M[i]);
		free(L[i]);
		free(U[i]);
		Sol[i] = H[i];
	}
	free(M);
	free(L);
	free(U);
	free(G);
	free(H);
}

void AMG_Direct_solver(sparse_matrix* L,sparse_matrix* U,double* F,double* Sol,int size){   // simple LU-factorization
	copy_vector_to(F,Sol,size);
	L_triang_invert(L,Sol);
	U_triang_invert(U,Sol);
	func_counter++;
}

int AMG_get_coarse_number(int* Coarsening,int size){
	int i;
	int n = 0;
	for (i=0;i<size;i++) if (Coarsening[i]==COARSE) n++;
	return n;
}

double AMG_recursive_solution(amg_system_info* Data,double* F,double* Sol,int depth){
	double* Copy_Sol;
	double* Fine_Residuum;
	double* Coarse_Residuum;
	double* Fine_Correction;
	double* Coarse_Correction;
	double s,r0,r1,r2;
	int k,dyn_iter,smooth_iter;
	int fine_size = Data->Fine_sizes[depth];
	int coarse_size = Data->Coarse_sizes[depth];
	int* Iter = (*(Data->Get_iter))(Data->depth);
	
	// pre-smoothing
	r0 = ILU_residuum(Data->Coarse_A[depth],Data->L[depth],Data->U[depth],Sol,F);
	if (Data->AMG_coarsening==AGGRESSIVE && depth==Data->depth-1){
		smooth_iter = Data->AMG_aggr_smoothing_iterations;
		AMG_smoother(Data->Coarse_A[depth],F,Sol,Data->Coarsening[depth],1.,UNDECIDED,1);
	}
	else{
		smooth_iter = Data->AMG_smoothing_iterations;
		AMG_smoother(Data->Coarse_A[depth],F,Sol,Data->Coarsening[depth],1.,UNDECIDED,smooth_iter);
	}
	dyn_iter = smooth_iter;
	r2 = r0;
	for (k=0;k<Iter[depth];k++){
		// compute residuum;
		Copy_Sol = clone_vector(Sol,fine_size);
		Fine_Residuum = clone_vector(F,fine_size);
		linear_map(Fine_Residuum,-1.,Data->Coarse_A[depth],Sol);
		r1 = r2;
		// restriction
		Coarse_Residuum = sparse_mult(Data->Restriction[depth],Fine_Residuum);
		Coarse_Correction = zero_vector(coarse_size);
		// coarse solution
		if (depth>1) AMG_recursive_solution(Data,Coarse_Residuum,Coarse_Correction,depth-1);
		else (*Data->AMG_coarsest_solver)(Data->L[0],Data->U[0],Coarse_Residuum,Coarse_Correction,Data->Fine_sizes[depth-1]);
		// prolongation
		Fine_Correction = sparse_mult(Data->Prolongation[depth],Coarse_Correction);
		s = scalar(Fine_Residuum,Fine_Residuum,fine_size)/sparse_bilinear(Fine_Residuum,
		 Data->Coarse_A[depth],Fine_Correction);
		vector_add(Sol,Fine_Correction,s,fine_size);
		// post-smoothing
		
		AMG_smoother(Data->Coarse_A[depth],F,Sol,Data->Coarsening[depth],SOR_alpha,COARSE,smooth_iter);
		//AMG_set_PCG_matrices(Data->Coarse_A[depth],Data->L[depth],Data->U[depth]);
		//pcg(Sol,F,Fine_sizes[depth],1E-8,1);
		AMG_smoother(Data->Coarse_A[depth],F,Sol,Data->Coarsening[depth],SOR_alpha,UNDECIDED,smooth_iter);
		// test residuum
		r2 = ILU_residuum(Data->Coarse_A[depth],Data->L[depth],Data->U[depth],Sol,F);
		if (r2/r1>=0.9 && r2/r0>1E-8 && dyn_iter/smooth_iter<2){
			copy_vector_to(Copy_Sol,Sol,fine_size);
			AMG_smoother(Data->Coarse_A[depth],F,Sol,Data->Coarsening[depth],1.,UNDECIDED,smooth_iter);
			r2 = ILU_residuum(Data->Coarse_A[depth],Data->L[depth],Data->U[depth],Sol,F);
			dyn_iter += smooth_iter;
			k--;
			if (AMG_info && depth>=Data->depth-1) printf("correction GS_iter %d residuum %f\n",dyn_iter,(double)r2/r1);
		}
		else{
			if (AMG_info && depth>=Data->depth-1) printf("iteration depth %d coarse size %d GS_iter %d residuum %f\n",depth,fine_size,dyn_iter,(double)r2/r1);
			dyn_iter = smooth_iter;
		}
		// clean
		free(Copy_Sol);
		free(Fine_Residuum);
		free(Coarse_Residuum);
		free(Fine_Correction);
		free(Coarse_Correction);
	}
	free(Iter);
	return r2/r0;
}

double AMG_full_multigrid(amg_system_info* Data,double* F,double* Sol){
	double* New_Sol;
	double* Interpol_Sol;
	double* S;
	double r,r0,s,q;
	int i;
	int max_depth = Data->depth;
	double** Coarse_F = (double**)malloc(max_depth*sizeof(double*));
	int* Iter = (*(Data->Get_iter))(max_depth);
	r0 = ILU_residuum(Data->Coarse_A[max_depth-1],Data->L[max_depth-1],Data->U[max_depth-1],Sol,F);
	func_counter = 0;
	
	// compute restricted F
	Coarse_F[max_depth-1] = clone_vector(F,Data->Fine_sizes[max_depth-1]);
	for (i=max_depth-1;i>0;i--){
		Coarse_F[i-1] = sparse_mult(Data->Restriction[i],Coarse_F[i]);
		Iter[i] = 1;
	}
	// obtain coarsest guess
	New_Sol = zero_vector(Data->Fine_sizes[0]);
	(*AMG_coarsest_solver)(Data->L[0],Data->U[0],Coarse_F[0],New_Sol,Data->Fine_sizes[0]);
	// initial V-cycles
	for (i=1;i<max_depth-1;i++){
		Interpol_Sol = sparse_mult(Data->Prolongation[i],New_Sol);
		AMG_recursive_solution(Data,Coarse_F[i],Interpol_Sol,i);
		New_Sol = (double*)realloc(New_Sol,Data->Fine_sizes[i]*sizeof(double));
		copy_vector_to(Interpol_Sol,New_Sol,Data->Fine_sizes[i]);
		free(Interpol_Sol);
	}
	//obtain finest guess
	Interpol_Sol = sparse_mult(Data->Prolongation[max_depth-1],New_Sol);
	copy_vector_to(Interpol_Sol,Sol,Data->Fine_sizes[max_depth-1]);
	S = clone_vector(F,Data->Fine_sizes[max_depth-1]);
	L_triang_invert(Data->L[max_depth-1],S);
	U_triang_invert(Data->U[max_depth-1],S);
	s = scalar(Interpol_Sol,S,Data->Fine_sizes[max_depth-1]);
	sparse_multiplication(Data->Coarse_A[max_depth-1],Interpol_Sol);
	L_triang_invert(Data->L[max_depth-1],Interpol_Sol);
	U_triang_invert(Data->U[max_depth-1],Interpol_Sol);
	q = scalar(Sol,Interpol_Sol,Data->Fine_sizes[max_depth-1]);
	scalar_mult(s/q,Sol,Data->Fine_sizes[max_depth-1]);
	
	AMG_smoother(Data->Coarse_A[max_depth-1],Coarse_F[max_depth-1],Sol,Data->Coarsening[max_depth-1],SOR_alpha,UNDECIDED,2);
	//normal AMG
	AMG_recursive_solution(Data,F,Sol,max_depth-1);
	r = ILU_residuum(Data->Coarse_A[max_depth-1],Data->L[max_depth-1],Data->U[max_depth-1],Sol,F);
	//clean 
	for (i=0;i<max_depth;i++) free(Coarse_F[i]);
	free(Coarse_F);
	free(Iter);
	free(New_Sol);
	free(Interpol_Sol);
	free(S);
	if (AMG_info) printf("number of graph leafs: %d\n",func_counter/Iter[max_depth-1]);
	return r/r0;
}

amg_system_info* AMG_setup(
		sparse_matrix* A,
		int deg_of_freedom,
		int depth,
		int smoothing_iterations,
		int aggr_smoothing_iterations,
		int coarsening,
		int cycle_structure,
		int prolongation_method,
		void (*AMG_coarsest_solver)(sparse_matrix* L,sparse_matrix* U,double* F,double* Sol,int size),
		int* (*Get_iter)(int depth)
		){
	int** Old_Tupels;
	int** New_Tupels;
	int** Strong_connections;
	int** Strong_transpose;
	int** Strong_positions;
	int* Old_Levels;
	int* New_Levels;
	int* Old_Nodes;
	int* New_Nodes;
	int i,j,c_type;
	int main_size = A->size;
	amg_system_info* Info = (amg_system_info*)malloc(sizeof(amg_system_info));
	Info->Coarse_A = (sparse_matrix**)malloc(depth*sizeof(sparse_matrix*));
	Info->Prolongation = (sparse_matrix**)malloc(depth*sizeof(sparse_matrix*));
	Info->Restriction = (sparse_matrix**)malloc(depth*sizeof(sparse_matrix*));
	Info->L = (sparse_matrix**)malloc(depth*sizeof(sparse_matrix*));
	Info->U = (sparse_matrix**)malloc(depth*sizeof(sparse_matrix*));
	Info->Coarsening = (int**)malloc(depth*sizeof(int*));
	Info->Coarse_sizes = (int*)malloc(depth*sizeof(int));
	Info->Fine_sizes = (int*)malloc(depth*sizeof(int));
	Info->AMG_deg_of_freedom = deg_of_freedom;
	Info->depth = depth;
	Info->AMG_smoothing_iterations = smoothing_iterations;
	Info->AMG_aggr_smoothing_iterations = aggr_smoothing_iterations;
	Info->AMG_coarsening = coarsening;
	Info->AMG_cycle_structure = cycle_structure;
	if (AMG_coarsest_solver!=NULL) Info->AMG_coarsest_solver = AMG_coarsest_solver;
	else Info->AMG_coarsest_solver = &AMG_Direct_solver;
	AMG_settings(deg_of_freedom,smoothing_iterations,aggr_smoothing_iterations,
	 coarsening,cycle_structure);
	if (Get_iter!=NULL) Info->Get_iter = Get_iter; else Info->Get_iter = &AMG_default_iterations;
	
	// prepare matrix
	Info->Coarse_A[depth-1] = clone(A);
	Info->Signs = make_positive_diag(Info->Coarse_A[depth-1]);
	Info->Fine_sizes[depth-1] = main_size;
	Old_Tupels = (int**)malloc(main_size*sizeof(int*));
	Old_Levels = (int*)malloc(main_size*sizeof(int));
	Old_Nodes = (int*)malloc(main_size*sizeof(int));
	AMG_init_finest_tupels(Old_Tupels,Old_Levels,Old_Nodes,main_size);
	if (AMG_info) printf("finest occupation: %f\n",degree_of_occupation(Info->Coarse_A[depth-1]));
	for (i=depth-1;i>0;i--){
		Info->L[i] = sparse_zero(Info->Fine_sizes[i]);
		Info->U[i] = sparse_zero(Info->Fine_sizes[i]);
		incomplete_LU_factorization(Info->Coarse_A[i],Info->L[i],Info->U[i]);
		// F/C - Splitting
		Strong_connections = (int**)malloc(Info->Fine_sizes[i]*sizeof(int*));
		Strong_positions = (int**)malloc(Info->Fine_sizes[i]*sizeof(int*));
		Strong_transpose = (int**)malloc(Info->Fine_sizes[i]*sizeof(int*));
		Info->Coarsening[i] = (int*)malloc(Info->Fine_sizes[i]*sizeof(int));
		if(Info->AMG_coarsening==AGGRESSIVE && i==depth-1){
			Info->Coarse_sizes[i] = aggressive_coarsening(Info->Coarse_A[i],Info->Coarsening[i],
			 Strong_connections,Strong_positions,Strong_transpose,Old_Levels);
		}
		else{
			Info->Coarse_sizes[i] = standard_coarsening(Info->Coarse_A[i],Info->Coarsening[i],
			Strong_connections,Strong_positions,Strong_transpose,Old_Levels,STAND_ALONE);
		}
		if (AMG_info) printf("size %d : %d\n",i-1,AMG_get_coarse_number(Info->Coarsening[i],Info->Fine_sizes[i]));
		//if (i==depth-1) AMG_fix_isolated(Coarse_A[i],Fine_F,Fine_Sol,Coarsening[i]);
		Info->Fine_sizes[i-1] = Info->Coarse_sizes[i];
		New_Tupels = (int**)malloc(Info->Coarse_sizes[i]*sizeof(int*));
		New_Levels = (int*)malloc(Info->Coarse_sizes[i]*sizeof(int));
		New_Nodes = (int*)malloc(Info->Coarse_sizes[i]*sizeof(int));
		AMG_set_tupels(Info->Coarsening[i],New_Levels,Old_Levels,New_Nodes,Old_Nodes,New_Tupels,
		 Old_Tupels,Info->Fine_sizes[i]);
		//if (i==depth-1) print_regular_coarsening(New_Nodes,Info->Coarsening[i],Info->Fine_sizes[i],40);
		// set restriction- and prolongation-matrix
		if (Info->AMG_coarsening==AGGRESSIVE && i==depth-1) c_type = AGGRESSIVE; else c_type = STANDARD;
		Info->Prolongation[i] = sparse_zero(Info->Fine_sizes[i]);
		set_prolongation_map(Info->Coarsening[i],Info->Coarse_A[i],Info->Prolongation[i],Strong_connections,
		 Strong_positions,Strong_transpose,Old_Tupels,Old_Levels,Old_Nodes,c_type);
		Info->Restriction[i] = get_AMG_transposed(Info->Prolongation[i],Info->Coarse_sizes[i]);
		sparse_approximate(Info->Prolongation[i],AMG_tol);
		sparse_approximate(Info->Restriction[i],AMG_tol);
		// Galerkin-Approximation
		Info->Coarse_A[i-1] = get_restricted_matrix(Info->Coarse_A[i],Info->Restriction[i],Info->Prolongation[i]);
		sparse_approximate(Info->Coarse_A[i-1],AMG_tol);
		if (AMG_info) printf("occupation: %f\n",degree_of_occupation(Info->Coarse_A[i-1]));
		
		//for (j=0;j<Coarse_sizes[i];j++) if (Coarsening[depth-1][New_Nodes[j]]!=ISOLATED) test_vector[New_Nodes[j]] += 1.; //debug !
		// clean
		for (j=0;j<Info->Fine_sizes[i];j++){
			free(Strong_connections[j]);
			free(Strong_positions[j]);
			free(Strong_transpose[j]);
			free(Old_Tupels[j]);
		}
		free(Strong_connections);
		free(Strong_positions);
		free(Strong_transpose);
		free(Old_Tupels);
		free(Old_Levels);
		free(Old_Nodes);
		Old_Tupels = New_Tupels;
		Old_Levels = New_Levels;
		Old_Nodes = New_Nodes;
	}
	for (j=0;j<Info->Fine_sizes[0];j++) free(Old_Tupels[j]);
	free(New_Tupels);
	free(New_Levels);
	free(New_Nodes);
	
	Info->L[0] = sparse_zero(Info->Fine_sizes[0]);
	Info->U[0] = sparse_zero(Info->Fine_sizes[0]);
	sparse_LU_factorization(Info->Coarse_A[0],Info->L[0],Info->U[0]);
	Info->Coarse_sizes[0] = 0;
	Info->Coarsening[0] = NULL;
	Info->Prolongation[0] = NULL;
	Info->Restriction[0] = NULL;
	return Info;
}

double AMG_solve(amg_system_info* Data,double* F,double* Fine_Sol){
	double* Fine_F;
	int (*Old_precon)(int n,double* R,double* Z);
	void (*Old_matmult)(int n,double* x,double* y);
	int old_iter;
	int depth = Data->depth;
	sparse_matrix* A = Data->Coarse_A[depth-1];
	int main_size = A->size;
	int* Iter = (*(Data->Get_iter))(depth);
	AMG_settings(Data->AMG_deg_of_freedom,Data->AMG_smoothing_iterations,Data->AMG_aggr_smoothing_iterations,
	 Data->AMG_coarsening,Data->AMG_cycle_structure);
	
	old_iter = PCG_get_precon_iter();
	Old_matmult = PCG_get_mat_mult();
	Old_precon = PCG_get_precon();
	PCG_set_mat_mult(&PCG_mult);
	PCG_set_precon(&PCG_precon,10);
	
	Fine_F = clone_vector(F,main_size);
	sparse_multiplication(Data->Signs,Fine_F);
	AMG_fix_isolated(A,Fine_F,Fine_Sol,Data->Coarsening[depth-1]);
	if (diag_zero(A)){
		printf("still a zero diagonal element after pivoting -> abort\n");
		exit(0);
	}
	// solution
	double rel = matrix_residuum(A,Fine_Sol,Fine_F);
	if (rel!=0){
		switch (Data->AMG_cycle_structure){
			case STAND_ALONE:
				func_counter = 0;
				rel = AMG_recursive_solution(Data,Fine_F,Fine_Sol,depth-1);
				if (AMG_info) printf("number of graph leafs: %d\n",func_counter/Iter[depth-1]);
				break;
			case FULL_AMG:
				rel = AMG_full_multigrid(Data,Fine_F,Fine_Sol);
				break;
			case AMG_PCG:
				/*PCG_set_AMG_system(A,L,U,Restriction,Prolongation,Coarse_A,C_L,C_U,
				 Coarsening,Coarse_sizes,Fine_sizes,depth);
				PCG_set_AMG_solver(&AMG_recursive_solution);
				PCG_set_mat_mult(&PCG_AMG_mat_mult);
				PCG_set_precon(&PCG_AMG_precon,10);
				pcg(Fine_Sol,Fine_F,main_size,1E-4,10);
				rel = get_pcg_residuum();
				PCG_reset();*/
				break;
			default: printf("Warning AMG: no solution method choosen -> skip\n");
		}
	}
	
	// clean
	free(Iter);
	PCG_set_mat_mult(Old_matmult);
	PCG_set_precon(Old_precon,old_iter);
	return rel;
 }

int* AMG_default_iterations(int depth){
	int i;
	int* Iter = (int*)malloc(depth*sizeof(int));
	for (i=0;i<depth;i++) Iter[i] = 1;
	Iter[depth-1] = 10;
	Iter[depth-2] = 2;
	return Iter;
}

void set_coarsening_strategy(int strategy){
	AMG_coarsening = strategy;
}

void AMG_set_deg_of_freedom(int n){
	AMG_deg_of_freedom = n;
}

void AMG_free_data(amg_system_info* Data){
	int i;
	for (i=0;i<Data->depth;i++){
		if (Data->Coarse_A[i]!=NULL) free_sparse(Data->Coarse_A[i]);
		if (Data->Prolongation[i]!=NULL) free_sparse(Data->Prolongation[i]);
		if (Data->Restriction[i]!=NULL) free_sparse(Data->Restriction[i]);
		if (Data->L[i]!=NULL) free_sparse(Data->L[i]);
		if (Data->U[i]!=NULL) free_sparse(Data->U[i]);
		if (Data->Coarsening[i]!=NULL) free(Data->Coarsening[i]);
	}
	free_sparse(Data->Signs);
	free(Data->Coarse_A);
	free(Data->Prolongation);
	free(Data->Restriction);
	free(Data->L);
	free(Data->U);
	free(Data->Coarsening);
	free(Data->Fine_sizes);
	free(Data->Coarse_sizes);
}
