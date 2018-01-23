
#include "FEM.h"

// global Variablen //////////////////////////////////////

extern mesh2D glob_mesh;
extern point2D** Dual_mesh;
extern double* Dual_areas;
//extern mesh2D** multi_mesh;
//extern interpolation_map2D interpol2D;
static sparse_matrix* System_matrix;
static sparse_matrix* Left_precon;
static sparse_matrix* Right_precon;
static int** Main_conditions;
static int** Var_blocks;
static int* Block_len;
static void** Sub_solvers;
static int block_number;
static int GMRES_control;
static int solver_mode;

static int (*Cond)(int j_vertex,int k_var);
static double (*F_dirichlet)(int i,int n,double t,double dt);
static double (*F_neumann)(int i,int n,double t,double dt);
static double (*F_robin)(int i, int n,double t,double dt);
static double (*F_init)(int i_vertex,int var_index);

static void (*init_System)(void* Params);

//void set_sub_solvers(void** Funcs,int number)

int var_number;
int extrapolation_order;
int smoothing_iteration_number;
int preconditioner_mode;
int Gauss_Seidel_maxiter;
char* Filename;
char* Directory;
static struct ITLIN_OPT* Solver_options;
static struct ITLIN_INFO* Solver_info;
FILE* Mem_log;
int FEM_error = 0;

// Code ///////////////////////////////////////////////////

void set_var_number(int var_num){
	var_number = var_num;
	set_var_number2D(var_num);
}

void set_system_matrix(sparse_matrix* A){
	System_matrix = A;
}

void set_solver_mode(int mode){
	solver_mode = mode;
}

void set_solver_prec(double prec){
	Solver_options->tol = prec;
}

void set_init_procedure(void (*F_init_System)(void* Params)){
	init_System = F_init_System;
}

double get_solver_prec(){
	return Solver_options->tol;
}

double get_var_number(){
	return var_number;
}

/*void copy_solution(int mesh_number,double** Source,double** Dest){
	int i,j,n;
	for (i=0;i<mesh_number;i++){
		n = multi_mesh[i]->size*var_number;
		for (j=0;j<n;j++){Dest[i][j] = Source[i][j];}
	}
}*/

void set_partial_solvers(void** Solvers){
	Sub_solvers = Solvers;
}

/*int** FEM_get_boundary_conditions(int mesh_index,int block_index){
	int i,j;
	int d = glob_mesh.size;
	int** Res = (int**)malloc(d*sizeof(int*));
	for (i=0;i<d;i++){
		Res[i] = (int*)malloc(Block_len[block_index]*sizeof(int));
		for (j=0;j<Block_len[block_index];j++) Res[i][j] = 
		Multi_conditions[mesh_index][i][j+Var_blocks[block_index][0]];
	}
	return Res;
}*/

/*void set_single_dirichlet(sparse_matrix* A,double* F,int var_index,int mesh_index,double value){
	int i = var_index*glob_mesh.size+mesh_index;
	if (i<A->size){
		reset_row(A,i);
		A->Values[i][0] = 1;
		F[i] = value;
	}
}*/

/*int** resctrict_conditions(int var,int varnum,int** conditions){
	int i,j;
	int n = glob_mesh.size;
	int ** Res = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++){
		Res[i] = (int*)malloc(varnum*sizeof(int));
		for (j=0;j<varnum;j++){
			Res[i][j] = conditions[i][var+j];
		}
	}
	return Res;
}*/

int** set_pure_conditions(int varnum,int cond){
	int i,j;
	int n = glob_mesh.size;
	int ** Res = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++){
		Res[i] = (int*)malloc(varnum*sizeof(int));
		if (glob_mesh.Is_boundary[i]>=0){
			for (j=0;j<varnum;j++){Res[i][j] = cond;}
		}
		else{
			for (j=0;j<varnum;j++){Res[i][j] = NONE;}
		}
	}
	return Res;
}

inline void free_cond(int** Cond){
	int i;
	for (i=0;i<glob_mesh.size;i++){free(Cond[i]);}
	free(Cond);
}

void delete_dirichlet_components(double* F,int** conditions,int size){
	int k,i_mesh,n_mesh;
	int d = glob_mesh.size;
	for (k=0;k<size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		if (conditions[i_mesh][n_mesh]==DIRICHLET) F[k] = 0;
	}
}

void set_dirichlet(sparse_matrix* A,double* F,int** conditions,double t,double dt,int zero_mode){
	double a;
	int j,k,l,i_mesh,n_mesh;
	int d = glob_mesh.size;
	for (k=0;k<A->size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		if (conditions[i_mesh][n_mesh]==DIRICHLET){
			reset_row(A,k);
			A->Values[k][0] = 1;
			if (zero_mode>=0){F[k] = (*F_dirichlet)(i_mesh,n_mesh,t,dt);}
			else{F[k] = 0;}
		}
		else{
			j = 0;
			while (j<A->Len[k]){
				l = A->Indices[k][j];
				i_mesh = l % d;
				n_mesh = l / d;
				if (conditions[i_mesh][n_mesh]==DIRICHLET){
					a = A->Values[k][j];
					remove_element_at(A,k,j);
					if (zero_mode>=0){F[k] -= a*(*F_dirichlet)(i_mesh,n_mesh,t,dt);}
					else{F[k] -= a*0;}					
				}
				else{j++;}
			}
		}
	}
}

void set_direct_dirichlet(sparse_matrix* A,double* F,int** conditions,double* Values){
	double a;
	int j,k,l,i_mesh,n_mesh;
	int d = glob_mesh.size;
	for (k=0;k<A->size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		if (conditions[i_mesh][n_mesh]==DIRICHLET){
			reset_row(A,k);
			A->Values[k][0] = 1;
			F[k] = Values[k];
		}
		else{
			j = 0;
			while (j<A->Len[k]){
				l = A->Indices[k][j];
				i_mesh = l % d;
				n_mesh = l / d;
				if (conditions[i_mesh][n_mesh]==DIRICHLET){
					a = A->Values[k][j];
					remove_element_at(A,k,j);
					F[k] -= a*Values[l];
				}
				else{j++;}
			}
		}
	}
}

sparse_matrix* FEM_get_dirichlet_correction(sparse_matrix* A,int** conditions, double diag){
	double a;
	int j,k,l,i_mesh,n_mesh;
	int d = glob_mesh.size;
	sparse_matrix* B = sparse_zero(A->size);
	for (k=0;k<A->size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		if (conditions[i_mesh][n_mesh]==DIRICHLET){
			reset_row(A,k);
			A->Values[k][0] = diag;
			B->Values[k][0] = diag;
		}
		else{
			j = 0;
			while (j<A->Len[k]){
				l = A->Indices[k][j];
				i_mesh = l % d;
				n_mesh = l / d;
				if (conditions[i_mesh][n_mesh]==DIRICHLET){
					a = A->Values[k][j];
					remove_element_at(A,k,j);
					insert_sparse(B,-a,k,l);
				}
				else{j++;}
			}
		}
	}
	return B;
}

void FEM_impose_dirichlet(double* Solution,int size,int** conditions,
 double (*F_dirichlet)(int i_mesh,int n_mesh,double t,double dt),double t,double dt){
	int i,n,k;
	int d = glob_mesh.size;
	for (k=0;k<size;k++){
		i = k % d;
		n = k / d;
		if (conditions[i][n]==DIRICHLET){ 
			if (F_dirichlet!=NULL) Solution[k] = (*F_dirichlet)(i,n,t,dt);
			else Solution[k] = 0;
		}
	}
}

int get_bound_num(){
	int i;
	int k = 0;
	for (i=0;i<glob_mesh.size;i++) if (is_boundary(i)>=0) k++;
	return k;
}

/*int set_const_bound(int** Cond,int var,int size,double* Sol,
	double* Old_Sol,double dt){
	double r;
	int j,k,i_mesh,n_mesh;
	int b = get_bound_num();
	int d = glob_mesh.size;
	double* List = (double*)malloc(size*b*sizeof(double));
	int* Indices = (int*)malloc(size*b*sizeof(int));
	j = 0;
	for (k=d*var;k<d*(var+size);k++){
		i_mesh = k % d;
		if (is_boundary(i_mesh)>=0){
			r = fabs((Sol[k]-Old_Sol[k])/dt);
			List[j] = r;
			Indices[j] = k;
			j++;
		} 
	}
	if (j!=b) printf("Fehler in set_const_bound: j=%d != size=%d\n",j,size*b);
	int counter = 0;
	double sigma = get_sqr_mean(List,size*b);
	for (k=0;k<size*b;k++){
		i_mesh = Indices[k] % d;
		n_mesh = Indices[k] / d;
		if (List[k]<sigma/4){
			Cond[i_mesh][n_mesh] = DIRICHLET;
			counter++;
		}
	}
	free(List);
	free(Indices);
	return counter;
}*/

void vector_dirichlet(double* F,int size,int** conditions){
	int k,i_mesh,n_mesh;
	int d = glob_mesh.size;
	for (k=0;k<size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		if (conditions[i_mesh][n_mesh]==DIRICHLET){F[k] = 0;}
	}
}

/*void set_neumann(sparse_matrix* A,double* F,int** conditions,int varnum,double t,double dt,int zero_mode){  //Achtung Konstanten und dt in F_neumann
	int i,n_mesh,i_mesh;
	int n = A->size;
	int d = glob_mesh.size;
	sparse_matrix* RB;
	sparse_matrix* Sum = sparse_zero(n);
	set_var_number2D(varnum);
	for (i=0;i<varnum;i++){
		RB = set_matrix_bound_Aij_00_2D(conditions,NEUMANN,i,i,&insert_A_aV);
		sparse_add(Sum,RB,1);
		free_sparse(RB);
	}
	double* Grad = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		i_mesh = i % d;
		n_mesh = i / d;
		if (zero_mode>=0){Grad[i] = (*F_neumann)(i_mesh,n_mesh,t,dt);}
		else {Grad[i] = 0;}
	}
	linear_map(F,1,Sum,Grad);
	free_sparse(Sum);
	free(Grad);
}*/

/*void set_zero_Neumann(sparse_matrix* A,double* F,int* Indices,double* Values,int len){
	int i_mesh,n_mesh,j,k,l;
	int d = glob_mesh.size;
	int c = get_center_index();
	double a = 0;
	for (k=0;k<A->size;k++){
		i_mesh = k % d;
		if (i_mesh!=c){
			j = 0;
			while (j<A->Len[k]){
				l = A->Indices[k][j];
				i_mesh = l % d;
				n_mesh = l / d;
				if (i_mesh==c && in_list(Indices,len,n_mesh)>=0){
					a = A->Values[k][j];
					remove_element_at(A,k,j);
					F[k] -= a*Values[n_mesh];
				}
				else{j++;}
			}
		}
		if (a!=0){break;}
	}
}*/

/*void set_robin(sparse_matrix* A,double* F,int** conditions,int varnum,double t,double dt){  //Achtung Konstanten und dt in F_neumann
	int i,n_mesh,i_mesh;  //f_robin(-1,i) muss alpha ergeben mit u_a =-alpha*(u-u_fix)na
	int n = A->size;
	int d = glob_mesh.size;
	sparse_matrix* RB;
	sparse_matrix* Sum = sparse_zero(n);
	for (i=0;i<varnum;i++){
		RB = set_matrix_bound_Aij_00_2D(conditions,ROBIN,i,i,&insert_A_aV);
		sparse_add(Sum,RB,(*F_robin)(-1,i,t,dt));	
		free_sparse(RB);		
	}
	sparse_add(A,Sum,1);
	double* U_fix = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		i_mesh = i % d;
		n_mesh = i / d;
		U_fix[i] = (*F_robin)(i_mesh,n_mesh,t,dt);
	}
	linear_map(F,1,Sum,U_fix);
	free_sparse(Sum);
	free(U_fix);
}*/

/*void boundary_conditions(sparse_matrix* A,double* F,int** conditions,double t,double dt){
	set_neumann(A,F,conditions,var_number,t,dt,NONE);
	set_robin(A,F,conditions,var_number,t,dt);
	set_dirichlet(A,F,conditions,t,dt,NONE);
}*/

/*void gauge(double* Sol,int var){
	int i;
	int d = glob_mesh.size;
	int size = get_bound_num();
	double* List = (double*)malloc(size*sizeof(double));
	int k = 0;
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		List[k] = Sol[var*d+i];
		k++;
	}
	double mean = get_mean(List,size);
	for (i=var*d;i<(var+1)*d;i++) Sol[i] -= mean;
	free(List);
}

double mean_gauge(double* Sol,int var){
	int i;
	int d = glob_mesh.size;
	double mean = 0;
	for (i=0;i<d;i++) mean += Sol[d*var+i];
	mean /= (double)d;
	for (i=0;i<d;i++) Sol[d*var+i] -= mean;
	return -mean;
}*/

FILE* init_file(char* Dir,char* Name,int var_num){
	Filename = Name;
	Directory = Dir;
	var_number = var_num;
	printf("output data: %s/%s\n",Directory,Filename);
	FILE* Res = open_file(Directory,Filename,"w+");
	return Res;
}

int Inv_matvec(int n,double* Solution,double* b){
	int i,j,diag;
	double sum;
	sparse_matrix* A = System_matrix;
	for (i=0;i<n;i++){
		diag = -1;
		sum = b[i];
		for (j=0;j<A->Len[i];j++){
			if (A->Indices[i][j]==i && A->Values[i][j]!=0){diag = j;}
			else{sum -= A->Values[i][j]*Solution[A->Indices[i][j]];}							
		}
		if (diag<0){
			printf("Diagonalelement %d gleich null: Gauss-Seidel nicht moeglich !",i);
			return -1;			
		}
		else{Solution[i] = sum/A->Values[i][diag];}					
	}
	return 1;
}

void matvec(int n,double* y,double* z){
	int i,j;
	double sum;
	sparse_matrix* A = System_matrix;
	for (i=0;i<n;i++){
		sum = 0;
		for (j=0;j<A->Len[i];j++){sum += A->Values[i][j]*y[A->Indices[i][j]];}
		z[i] = sum;
	}
}

void preconl(int n,double* z, double* w){
	int i,j,m;
	double sum;
	int* Ind;
	double* Val;
	for (i=0;i<n;i++){
		sum = z[i];
		m = Left_precon->Len[i]-1;
		Ind = Left_precon->Indices[i];
		Val = Left_precon->Values[i];
		for (j=0;j<m;j++){
			sum -= Val[j]*w[Ind[j]];
		}
		sum /= Val[m];
		w[i] = sum;
	}
}

void preconr(int n,double* z, double* w){
	int i,j,m;
	double sum,d;
	int* Ind;
	double* Val;
	for (i=n-1;i>=0;i--){
		sum = z[i];
		m = Right_precon->Len[i]-1;
		Ind = Right_precon->Indices[i];
		Val = Right_precon->Values[i];
		d = (*Val);
		Val++;
		Ind++;
		for (j=1;j<=m;j++){
			//sum -= Val[j]*w[Ind[j]];
			sum -= (*Val)*w[*Ind];
			Val++;
			Ind++;
		}		
		sum /= d;
		w[i] = sum;
	}
}

int preconlr(int n,double *z,double* w){
	double* y = clone_vector(z,n);
	preconl(n,z,y);
	preconr(n,y,w);
	free(y);
	return 1;
}

int gauss_seidel_precon(int n,double* z,double* w){
	return gauss_seidel(&Inv_matvec,&matvec,w,z,System_matrix->size,-Gauss_Seidel_maxiter,5E-2);
}

void GMRES_iteration_control(int iter){
	double fac = (double)4/3;
	int max_count = 2;
	int n = iter / Solver_options->i_max;
	if (n>1) GMRES_control++;
	else if (n<=1) GMRES_control--;
	printf("CONTROL = %d\n",GMRES_control);
	if (GMRES_control>=max_count){
		Solver_options->i_max = (int)floor((double)fac*Solver_options->i_max);
		GMRES_control = 0;
		printf("setze i_max = %d\n",Solver_options->i_max);
	}
	if (GMRES_control<=-max_count){
		Solver_options->i_max = (int)floor((double)Solver_options->i_max/fac);
		GMRES_control = 0;
		printf("setze i_max = %d\n",Solver_options->i_max);
	}
}

void init_solver(int i_max,int maxiter,int GS_maxiter,double tol){
	Solver_options = (struct ITLIN_OPT*)malloc(sizeof(struct ITLIN_OPT));
	Solver_options->tol = tol;
	Solver_options->i_max = i_max;  
	Solver_options->maxiter = maxiter;
	Gauss_Seidel_maxiter = GS_maxiter;
	Solver_options->termcheck = CheckOnRestart;
	
	Solver_options->errorlevel = Verbose;
	Solver_options->monitorlevel = None;
	Solver_options->datalevel = None;
	
	Solver_options->errorfile = stdout;
	Solver_options->monitorfile = stdout;
	Solver_options->datafile = NULL;
	Solver_options->iterfile = NULL;
	Solver_options->resfile = NULL;
	Solver_options->miscfile = NULL;
	
	Solver_info = (struct ITLIN_INFO*)malloc(sizeof(struct ITLIN_INFO));
	
	preconditioner_mode = MATRIX;
	GMRES_control = 0;
	Left_precon = NULL;
	Right_precon = NULL;
} 

void set_var_blocks(int** (*get_var_blocks)(int** Len,int blocks),int blocks){
	int i;
	int** Block_sizes = (int**)malloc(sizeof(int*));
	Var_blocks = (*get_var_blocks)(Block_sizes,blocks);
	Block_len = Block_sizes[0];
	block_number = blocks;
	int sum = 0;
	for (i=0;i<blocks;i++){sum += Block_len[i];}
	set_var_number(sum);
	printf("simulation with %d variables divided into %d blocks\n",sum,blocks);
}

void set_boundary_equations(
int (*cond)(int j_vertex,int k_var),
double (*f_dirichlet)(int i,int n,double t,double dt),
double (*f_neumann)(int i,int n,double t,double dt),
double (*f_robin)(int i,int n,double t,double dt)){
	F_dirichlet = f_dirichlet;
	F_neumann = f_neumann;
	F_robin = f_robin;
	Cond = cond;
}

void set_initial_equations(double (*f_init)(int i_vertex,int var_index)){
	F_init = f_init;
}

void set_boundary_conditions(){
	int j,k,n;
	n = glob_mesh.size;
	Main_conditions = (int**)malloc(n*sizeof(int*));
	for (j=0;j<n;j++){
		Main_conditions[j] = (int*)malloc(var_number*sizeof(int));
		for (k=0;k<var_number;k++){
			Main_conditions[j][k] = (*Cond)(j,k);
		}
	}
}

void set_preconditioner(sparse_matrix* A,int mode){
	int n = A->size;
	if (Left_precon!=NULL) free_sparse(Left_precon);
	if (Right_precon!=NULL) free_sparse(Right_precon);
	switch(mode){
		case NONE: 
			Left_precon = sparse_identity(n);
			Right_precon = sparse_identity(n);
			break;
		case DIAGONAL:
			Left_precon = sparse_zero(n);
			Right_precon = sparse_zero(n);
			diagonal_preconditioner(A,Left_precon,Right_precon);
			break;
		case ILU:
			Left_precon = sparse_zero(n);
			Right_precon = sparse_zero(n);
			incomplete_LU_factorization(A,Left_precon,Right_precon);
			break;
	}
}

double** set_initial_data(char* filename,double* Init_time,int* Init_step){
	int j,k,n;
	FILE* File;
	if (strcmp(filename,"none")==0){
		File = NULL;
	}
	else{
		char fullname[512];
		sprintf(fullname,"%s/data",filename);
		File = fopen(fullname,"r");
	}
	if (File==NULL){
		n = glob_mesh.size;
		double** Init = (double**)malloc(2*sizeof(double*));
		Init[0] = (double*)malloc(var_number*n*sizeof(double));
		Init[1] = (double*)malloc(var_number*n*sizeof(double));
		for (j=0;j<n;j++){
			for (k=0;k<var_number;k++){
				Init[0][k*n+j] = (*F_init)(j,k);
				Init[1][k*n+j] = Init[0][k*n+j];
			}
		}
		*Init_step = 0;
		return Init;
	}
	else{
		printf("load initil data from %s ...\n",filename);
		*Init_time = get_last_time(File,&n,&k,Init_step);
		double** Res = get_last_step_data(File,n,k,*Init_step);
		fseek(File,0,SEEK_END);
		return Res;
	}
}

void set_boundary_equation(sparse_matrix* A,double* b,int equ,int var,double factor){
	int i;
	int d = glob_mesh.size;
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		if (b!=NULL) b[equ*d+i] = 0;
		if (A!=NULL){
			reset_row(A,equ*d+i);
			insert_sparse(A,1.,equ*d+i,equ*d+i);
			insert_sparse(A,-factor,equ*d+i,var*d+i);
		}
	}
}

/*int cmp_sol(int mesh_number,double*** Solutions){
	int i,j,n,m;
	m = extrapolation_order-1;
	for (i=0;i<mesh_number;i++){
		n = multi_mesh[i]->size;
		for (j=0;j<n;j++){
			if (fabs(Solutions[m][i][j]-Solutions[m][i][j+n])>1E-3){
				return -1;
			}
		}
	}
	return 1;
}*/

void main_solve(int steps,double time_length,FILE* file,int interval,char* init_file){
	int i,n,m;
	double t;
	double t0 = 0;
	double* Old_solution;
	double* Solution;
	double** Data;
	double dt = (double)time_length/(steps-1);
	int init_steps = 0;
	
	if (file!=NULL){
		set_boundary_conditions();
		(*init_System)(NULL);
		Data = set_initial_data(init_file,&t0,&init_steps);
		Old_solution = Data[0];
		Solution = Data[1];
		t0 = 0;							// take out when continuing a simulation !
		init_steps = 0;					// take out when continuing a simulation !
		t = t0;   
		//clock_t start = clock();
		n = glob_mesh.size;
		printf("Starte Rechnung\n");
		//Old_solution = clone_vector(Solution,n*var_number);
		fprintf(file,"                                       \n");
		partial_decoupling_solver(Solution,Old_solution,t0,0);
		append_data(file,Solution,n,var_number,t0);
		
		for (i=1;i<=steps;i++){
			t = t0+(double)time_length*i/steps;
			partial_decoupling_solver(Solution,Old_solution,t,dt);
			if (FEM_error){
				printf("Error in FEM -> abort computation\n");
				break;
			}
			if (i % interval == 0){append_data(file,Solution,n,var_number,t);}
			if (steps/100>0 && i % (steps/100)== 0){
				m = 100*i/steps;
				printf("\r%d%% abgeschlossen",m);
				fflush(stdout);
			}
		}
		//clock_t end = clock();
		rewind(file);
		fprintf(file,"%d %d %d",glob_mesh.size,var_number,init_steps+i/interval);
		fclose(file);
		free_conditions();
		//if (Left_precon!=NULL){free_sparse(Left_precon);}
		//if (Right_precon!=NULL){free_sparse(Right_precon);}
		free(Solver_options);
		free(Solver_info);
		printf("\nRechnung fertig\n");
		//printf("Rechenzeit: %fs\n",(double)(end-start)/CLOCKS_PER_SEC);
	}
	else{printf("Keine Outputdatei vorhanden\n");}
}

int solver(double* F,double* Sol,int (*Precon)(int n,double *z,double* w)){
	double* H;
	int iter = 0;
	int n = System_matrix->size;
	if (Precon==preconlr){set_preconditioner(System_matrix,ILU);}
	switch(solver_mode){
#ifndef __cplusplus
	    case GMRES:	
			H = clone_vector(Sol,n);					
			gmres(n,H,&matvec,Precon,NULL,F,Solver_options,Solver_info);			
			(*Precon)(n,H,Sol);		
			free(H);
			iter = Solver_info->iter;
			//GMRES_iteration_control(iter);
			break;
#endif
		case GAUSS_SEIDEL:
			iter = gauss_seidel(&Inv_matvec,&matvec,Sol,F,n,smoothing_iteration_number,Solver_options->tol);
			break;
		case PCG: 	
			PCG_set_precon(Precon);
			PCG_set_mat_mult(&matvec);
			iter = pcg(Sol,F,n,Solver_options->tol,Solver_options->maxiter);
			if (iter<0){
				printf("versuche ILU-preconcitioning\n");
				set_preconditioner(System_matrix,ILU);
				PCG_set_precon(&preconlr);
				iter = pcg(Sol,F,n,Solver_options->tol,Solver_options->maxiter);
			}
			Solver_info->iter = iter;
			break;
	}
	return iter;
}

void partial_decoupling_solver(double* Solution,double* Old_solution,double t,double dt){
	int i;
	void (*Operator)(double* Solution,double* Old_solution,double t,double dt);
	FEM_impose_dirichlet(Solution,var_number*glob_mesh.size,Main_conditions,F_dirichlet,t,dt);
	for (i=0;i<block_number;i++){
		Operator = (OP_type)Sub_solvers[i];
		if (Operator!=NULL){
			(*Operator)(Solution,Old_solution,t,dt);
		}
	}
}

int FEM_solver(double* F,double* Sol,...){
	va_list args;
	int iter = 0;
	int (*Precon)(int n,double *z,double* w) = NULL;
	void (*Matvec)(int n,double* Solution,double* b) = NULL;
	int (*Inv_Matvec)(int n,double* Solution,double* b) = NULL;
	int n = System_matrix->size;
	va_start(args,Sol);
	switch(solver_mode){
#ifndef __cplusplus
		case GMRES:
			Precon = (PRECON_type)va_arg(args,void*);
			if (Precon!=NULL){
				set_preconditioner(System_matrix,ILU);
				double* H = clone_vector(Sol,n);					
				gmres(n,H,&matvec,Precon,NULL,F,Solver_options,Solver_info);			
				(*Precon)(n,H,Sol);		
				free(H);
				iter = Solver_info->iter;
			}
			break;
#endif
		case GAUSS_SEIDEL:
			Inv_Matvec = (PRECON_type)va_arg(args,void*);
			Matvec = (MULT_type)va_arg(args,void*);
			if (Matvec!=NULL && Inv_Matvec!=NULL){
				iter = gauss_seidel(&Inv_matvec,&matvec,Sol,F,n,smoothing_iteration_number,Solver_options->tol);
			}
			break;
		case PCG:
			//Matvec = va_arg(args,void*);
			Precon = (PRECON_type)va_arg(args,void*);
			if (Precon!=NULL){
				PCG_set_precon(Precon);
				PCG_set_mat_mult(&matvec);
				iter = pcg(Sol,F,n,Solver_options->tol,Solver_options->maxiter);
				Solver_info->iter = iter;
			}
			else{
				printf("Fehler: Kein Preconditioner oder Matrix-Multiplikator gewählt !-> abbruch\n");
				exit(0);
			}
			break;
		case ILU_PCG:
			Matvec = (MULT_type)va_arg(args,void*);
			Left_precon = va_arg(args,sparse_matrix*);
			Right_precon = va_arg(args,sparse_matrix*);
			if (Matvec!=NULL && Left_precon!=NULL && Right_precon!=NULL){
				PCG_set_mat_mult(Matvec);
				PCG_set_precon(&preconlr);
				iter = pcg(Sol,F,n,Solver_options->tol,Solver_options->maxiter);
				Solver_info->iter = iter;
			}
			break;
	}
	va_end(args);
	return iter;
}

double get_adaptive_time_step(double dt,sparse_matrix* A,double* X){
	int i,j;
	double a,sum;
	int n = A->size;
	double min = dt;
	for (i=0;i<n;i++){
		sum = 0;
		for (j=0;j<A->Len[i];j++) sum += fabs(A->Values[i][j]*X[A->Indices[i][j]]);
		a = X[i]/sum;
		if (a<min) min = a;
	}
	return min/10.;
}

#ifndef __cplusplus

void power_inversion(double* X,sparse_matrix* A,int power){
	int i,j;
	int n = A->size;
	set_solver_mode(ILU_PCG);
	set_system_matrix(A);
	set_solver_prec(1E-12);
	double* Y = clone_vector(X,n);
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(A,L,U);
	double r0 = matrix_residuum(A,Y,X);
	if (r0>0){
		for (i=0;i<power;i++){
			FEM_solver(X,Y,&matvec,L,U);
			for (j=0;j<n;j++) X[j] = Y[j];
		}
	}
	free(Y);
	set_solver_prec(1E-8);
}

void adaptive_inversion(double* X,sparse_matrix* A,double initial_dt){
	int j;
	double dt,r0;
	double* Y;
	sparse_matrix* B;
	sparse_matrix* L;
	sparse_matrix* U;
	int n = A->size;
	sparse_matrix* E = sparse_identity(n);
	set_solver_mode(ILU_PCG);
	set_solver_prec(1E-12);
	double s = 0;
	do{
		dt = get_adaptive_time_step(initial_dt,A,X);
		if (dt>=initial_dt) dt = initial_dt; else printf("set timestep to %f\n",dt);
		B = clone(E);
		L = sparse_zero(n);
		U = sparse_zero(n);
		sparse_add(B,A,dt);
		set_system_matrix(B);
		incomplete_LU_factorization(B,L,U);
		Y = clone_vector(X,n);
		r0 = matrix_residuum(B,Y,X);
		if (r0>0) FEM_solver(X,Y,&matvec,L,U);
		for (j=0;j<n;j++) X[j] = Y[j];
		s += dt;
		free(Y);
		free_sparse(B);
		free_sparse(L);
		free_sparse(U);
	}while(s<initial_dt);
	printf("adaptive solver: finished\n");
	free_sparse(E);
	set_solver_prec(1E-8);
}

#endif

double extrapolate(double*** Solutions,int mesh_index,int sol_index,double t,double dt){
	double u0 = Solutions[extrapolation_order-3][mesh_index][sol_index];
	double u1 = Solutions[extrapolation_order-2][mesh_index][sol_index];
	double u2 = Solutions[extrapolation_order-1][mesh_index][sol_index];
	double t0 = t-2*dt;
	double t1 = t-dt;
	double t2 = t;
	double t3 = t+dt;
	double detF = (t0-t1)*(t0-t2)*(t1-t2);
	double a = (u2*(t0-t1)+u0*(t1-t2)+u1*(t2-t0))/detF;
	double b = (u2*(-t0*t0+t1*t1)+u1*(t0*t0-t2*t2)+u0*(-t1*t1+t2*t2))/detF;
	double c = (u2*t0*(t0-t1)*t1+t2*(u0*t1*(t1-t2)+u1*t0*(-t0 +t2)))/detF;
	return a*t3*t3+b*t3+c;
}

/*void extrapolation(double*** Solutions,int mesh_number,double t,double dt){
	int i,j,k,n;
	double** Guess = (double**)malloc(mesh_number*sizeof(double*));
	for (i=0;i<mesh_number;i++){
		n = multi_mesh[i]->size*var_number;
		Guess[i] = (double*)malloc(n*sizeof(double));
		for (j=0;j<n;j++){Guess[i][j] = extrapolate(Solutions,i,j,t,dt);}
	}
	for (k=0;k<extrapolation_order-1;k++){
		copy_solution(mesh_number,Solutions[k+1],Solutions[k]);
	}
	copy_solution(mesh_number,Guess,Solutions[extrapolation_order-1]);
	for (i=0;i<mesh_number;i++){free(Guess[i]);}
	free(Guess);
}*/

/*void multi_solver(double* Solution,double* F,int mesh_number,int varnum,
sparse_matrix* (*set_matrix)(double* F,int mesh_index,int mesh_size)){
	int i,n;	
	vectors = (double**)malloc(mesh_number*sizeof(double*));
	matrices = (sparse_matrix**)malloc(mesh_number*sizeof(sparse_matrix*));		
	for (i=mesh_number-1;i>=0;i--){	
		set_mesh(i);	
		n = glob_mesh.size;				
		if (i<mesh_number-1){			
			vectors[i] = zero_vector(multi_mesh[i]->size*varnum);
			interpolation2D(vectors[i],vectors[i+1],i,varnum,DOWN);
		}
		else{			
			vectors[i] = clone_vector(F,multi_mesh[i]->size*varnum);
		}											
		matrices[i] = (*set_matrix)(vectors[i],i,n);  						
	}	
	set_equations(matrices,vectors,mesh_number);
	Multigrid_solve(Solution,40,4,1E-3,varnum);			
	for (i=0;i<mesh_number;i++){
		free_sparse(matrices[i]);
		free(vectors[i]);
	}	
	free(matrices);
	free(vectors);	
}*/



int subdomain_solver(sparse_matrix* Matrix,double* Vector,double* Solution,int* Domain_list,int domain_size){
	int iter;
	int N = Matrix->size;
	int* Index_map = (int*)malloc(N*sizeof(int));
	sparse_matrix* Sub_matrix = restrict_matrix_rows(Matrix,Domain_list,domain_size,Index_map);
	double* Bound = restrict_matrix_cols(Sub_matrix,Solution,Index_map);
	double* Sub_vector = restrict_vector(Vector,Index_map,N,N-domain_size);
	vector_add(Sub_vector,Bound,1,N-domain_size);
	double* Sub_solution = restrict_vector(Solution,Index_map,N,N-domain_size);
	solver_mode = PCG;
	preconditioner_mode = MATRIX;
	System_matrix = Sub_matrix;
	iter = solver(Sub_vector,Sub_solution,&preconlr);
	map_back(Solution,Sub_solution,N,Index_map);
	
	free_sparse(Sub_matrix);
	free(Sub_vector);
	free(Sub_solution);
	free(Bound);
	return iter;
}

int get_domain_list(int** Lists,int list_index,int domain_number,int matrix_size,double* Solution){
	int k,i_mesh,n_mesh,cond;
	double y;
	int d = glob_mesh.size;
	//int c = get_center_index();
	Lists[list_index] = (int*)malloc(0);
	int size = 0;
	for (k=0;k<matrix_size;k++){
		i_mesh = k % d;
		n_mesh = k / d;
		y = glob_mesh.Points[i_mesh].y;
		//r = dist(&glob_mesh.Points[c],&glob_mesh.Points[i_mesh]);
		//if (n_mesh!=0 || r>0.1){
		switch(list_index){
			case 0: cond = (y<0);break;
			case 1: cond = (y>=0);break;
			default: cond = 0;
		}
		if (cond){
			Lists[list_index] = (int*)realloc(Lists[list_index],(size+1)*sizeof(int));
			Lists[list_index][size] = k;
			size++;
		}
	}
	return size;
}

void domain_decomposition_solver(sparse_matrix* Matrix,double* Vector,double* Solution,
	int domain_iterations,int domain_number){
	int i,k;
	double r,r0;
	double* Old;
	int* Sizes = (int*)malloc(domain_number*sizeof(int*));
	int** Lists = (int**)malloc(domain_number*sizeof(int*));
	for (i=0;i<domain_number;i++){
		Sizes[i] = get_domain_list(Lists,i,domain_number,Matrix->size,Solution);
	}
	for (k=0;k<domain_iterations;k++){
		Old = clone_vector(Solution,Matrix->size);
		r0 = euklid_norm(Old,Matrix->size);
		for (i=0;i<domain_number;i++){
			subdomain_solver(Matrix,Vector,Solution,Lists[i],Sizes[i]);
		}
		r = vec_dist(Solution,Old,Matrix->size);
		printf("iteration %d: r=%f\n",k+1,r/r0);
		free(Old);
	}
}

/*int sub_solver(double* Solution,double* Old_solution,int mesh_number,double t,double dt,
void (*Method)(sparse_matrix* matrix,double* vector,double* Prev,double* Prev_old,double t,double dt)){
	int i,list_size;
	int* List;
	double* H;
	double* Vector;
	double* Sub_vector;
	double* Sub_solution;	
	sparse_matrix* Sub_matrix;
	set_mesh(mesh_number-1);	
	int n = glob_mesh.size;	
	int N = Multi_matrix[mesh_number-1]->size;
	Conditions = Multi_conditions[mesh_number-1];
	choose_linear_matrix(mesh_number,mesh_number-1);
	Nonlinear_part = (*F_nonlinear_part)(N,var_number,Solution,Old_solution,t,dt);
	Next_nonlinear_part = (*F_next_nonlinear_part)(N,var_number,Solution,Old_solution,t,dt);	
	
	Vector = zero_vector(N);
	sparse_matrix* Matrix = sparse_zero(N);
	(*Method)(Matrix,Vector,Solution,Old_solution,t,dt); 
	solver_mode = PCG;
	preconditioner_mode = MATRIX;
	
	for (i=0;i<block_number;i++){
		if (i==0 && F_convection!=NULL) (*F_convection)(Solution,Old_solution,t,dt);
		List = generate_list(Var_blocks[i],Block_len[i],n,N);
		list_size = N-Block_len[i]*n;
		int* Index_map = (int*)malloc(N*sizeof(int));
		Sub_matrix = restrict_matrix_rows(Matrix,List,list_size,Index_map);
		H = restrict_matrix_cols(Sub_matrix,Solution,Index_map);
		Sub_vector = restrict_vector(Vector,Index_map,N,N-list_size);
		vector_add(Sub_vector,H,1,N-list_size);
		System_matrix = Sub_matrix;
		Sub_solution = restrict_vector(Solution,Index_map,N,N-list_size);
		
		switch(i){
			case 0: solver(Sub_vector,Sub_solution,&gauss_seidel_precon);
					if (Solver_info->iter>=Solver_options->maxiter-1){
						printf("versuche ILU-preconditioning\n");
						solver(Sub_vector,Sub_solution,&preconlr);
						map_back(Solution,Sub_solution,N,Index_map);
					}
					break;
			case 2: if (F_pressure_correction!=NULL){
						(*F_pressure_correction)(Solution,Old_solution,N,t,dt);
					}
					break;
			case 3: if (F_velocity!=NULL){
						(*F_velocity)(Solution,Old_solution,t,dt);
					}
					break;
			default:solver(Sub_vector,Sub_solution,&preconlr);
					map_back(Solution,Sub_solution,N,Index_map);
					break;
		}
		
		printf("iteration block %d: %d\n",i,Solver_info->iter);
		free(List);
		free(Index_map);
		free(Sub_vector);
		free(Sub_solution);
		free(H);
		free_sparse(Sub_matrix);
	}
	free(Vector);
	free_sparse(Matrix);
	Nonlinear_part = free_sparse(Nonlinear_part);
	Next_nonlinear_part = free_sparse(Next_nonlinear_part);
	return -1;
}*/

/*int simple_solver(double* Solution,double* Old_solution,int mesh_number,double t,double dt,
void (*Method)(sparse_matrix* matrix,double* vector,double* Prev,double t,double dt)){
	int iter;		
	set_mesh(mesh_number-1);	
	//int n = glob_mesh.size;	
	int N = Multi_matrix[mesh_number-1]->size;
	double* Vector = zero_vector(N);
	sparse_matrix* Matrix = sparse_zero(N);
	Conditions = Multi_conditions[mesh_number-1];	
	choose_linear_matrix(mesh_number,mesh_number-1);
	Nonlinear_part = (*F_nonlinear_part)(N,var_number,Solution,Old_solution,t,dt);
	Next_nonlinear_part = (*F_next_nonlinear_part)(N,var_number,Solution,Old_solution,t,dt);	
	(*Method)(Matrix,Vector,Solution,t,dt); 		
	
	solver_mode = GMRES;
	preconditioner_mode = MATRIX;
	System_matrix = Matrix;	
	iter = solver(Vector,Solution,&preconlr);
	
	free_sparse(Nonlinear_part);
	free_sparse(Next_nonlinear_part);	
	free_sparse(Matrix);
	free(Vector);
	
	return iter;
}*/
	

/*void free_globals(){
	free_sparse(Left_precon);			
	free_sparse(Right_precon);			
	free(Solver_options);
	free(Solver_info);
}*/

double* get_function(double* Sol,int mesh_size,int varnum,int equ,double (*Func)(double* X,int k)){
	int i,j;
	double* P = (double*)malloc(varnum*sizeof(double));
	double* Res = zero_vector(varnum*mesh_size);
	for (i=0;i<mesh_size;i++){
		if (Sol!=NULL) for (j=0;j<varnum;j++){P[j] = Sol[j*mesh_size+i];}
		Res[equ*mesh_size+i] = (*Func)(P,i);
	}
	free(P);
	return Res;
}

double* get_matrix_function(double* Sol,int mesh_size,int varnum,int equ,	// xx->0 xy->1 yx->2 yy->3
 double (*Func)(double* X,int k_mesh,int k_index)){
	int i,j;
	double* P = (double*)malloc(varnum*sizeof(double));;
	double* Res = zero_vector(varnum*mesh_size);
	for (i=0;i<mesh_size;i++){
		if (Sol!=NULL) for (j=0;j<varnum;j++){P[j] = Sol[j*mesh_size+i];}
		for (j=0;j<4;j++) Res[(equ+j)*mesh_size+i] = (*Func)(P,i,j);
	}
	free(P);
	return Res;
}

/*void get_spherical_Fourier(double* Data,int varnum,int size,double** a,double** b){
	int i,j,trig;
	
	double Tscheb(double* X,int k){
		return Tschebyscheff_disc(k,i,j,trig);
	}
	
	int n = glob_mesh.size;
	double* Base_function;
	for (i=0;i<n;i++){
		a[i] = (double*)malloc(2*sizeof(double));
		b[i] = (double*)malloc(2*sizeof(double));
	}
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			trig = COS;
			Base_function = get_function(NULL,n,0,0,&Tscheb);
			trig = SIN;
		}
	}
}*/


// Testfunktionen ////////////////////////////////////////////////////

void set_test_matrix(int n){
	int i;
	sparse_matrix* A = sparse_zero(n);
	insert_sparse(A,1,0,0);
	insert_sparse(A,1,n-1,n-1);
	for (i=1;i<n-1;i++){
		insert_sparse(A,2,i,i);
		insert_sparse(A,-1,i,i-1);
		insert_sparse(A,-1,i,i+1);
	}
	insert_sparse(A,-0.5,10,50);
}

double* set_test_vector(int n){
	int i;
	double* Res = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++){
		Res[i] = (double)i/n;
	}
	return Res;
}

void test_mem(){
	printf("memory in use: %d\n",mallinfo().arena);
	printf("or: %d\n",mallinfo().uordblks);
}

void init_log_file(){
	//Mem_log = fopen("/home/radszuweit/C-Programme/FEM2D/mem_log","w");
	//MALLOC_TRACE = "/home/radszuweit/C-Programme/FEM2D/mem_log";
	putenv("MALLOC_TRACE=~/C-Programme/FEM/FEM2D/mem_log");
	mtrace();
}

void disable_log_file(){
	muntrace();
}

// Ende Testfunktionen /////////////////////////////////////

void read_node_file2D(char* path,char* Name,mesh2D* Mesh,double** Attr,int* attr_num,int quiet){
	int i,j,k,bound,at;
	double x,y;
	int len = 0;
	/*char buff[256] = "";
	char name[256] = "";*/
	char* buff = (char*)malloc(256*sizeof(char));
	char* name = (char*)malloc(256*sizeof(char));
	strcpy(name,Name);
	strcat(name,".node");
	FILE* file = open_file(path,name,"r");
	readline(file,buff);
	char** line = split(buff," ",&len);
	if (len!=4){printf("Fehlerhaftes node-File\n");}
	else{
		Mesh->size = atoi(line[0]);
		Mesh->Points = (point2D*) malloc(Mesh->size*sizeof(point2D));
		Mesh->Is_boundary = (int*)malloc(Mesh->size*sizeof(int));
		at = atoi(line[2]);
		if (attr_num!=NULL) *attr_num = at;
		if (Attr!=NULL && at>0) *Attr = zero_vector(at*(Mesh->size)*sizeof(double));
		if (!quiet){
			printf("%s nodes\n",line[0]);
			printf("dimension: %s\n",line[1]);
			printf("attributes: %s\n",line[2]);
		}
		for (j=0;j<len;j++){free(line[j]);}
		free(line);
		int counter = 0;
		for (i=0;i<Mesh->size;i++){
			readline(file,buff);
			line = split(buff," ",&len);
			k = atoi(line[0]);
			x = atof(line[1]);
			y = atof(line[2]);
			if (Attr!=NULL){
				for (j=0;j<at;j++) (*Attr)[(Mesh->size)*j+i] = atof(line[j+3]);			
			}
			bound = atoi(line[at+3]);
			Mesh->Points[k] = init_point2D(x,y);
			if (bound==1){
				Mesh->Is_boundary[k] = 1;
				counter++;
				}
			else{Mesh->Is_boundary[k] = -1;}
			for (j=0;j<len;j++){free(line[j]);}
			free(line);
		}
		if (!quiet) printf("%d boundary nodes\n",counter);
	}
	free(buff);
	free(name);
	fclose(file);
}

index2D search_node(point2D* P1,point2D* P2,int** Taken){
	int i,j;
	double dmin,d;
	index2D res;
	point2D* Q = clone_point(P1);
	vec_add_mult(Q,P2,1.);
	vec_mult(Q,0.5);
	res.i = 0;
	dmin = dist(Q,&glob_mesh.Points[0]);
	for (i=1;i<glob_mesh.size;i++){
		d = dist(Q,&glob_mesh.Points[i]);
		if (d<dmin){
			dmin = d;
			res.i = i;
		}
	}
	res.j = 0;
	dmin = dist(Q,&glob_mesh.Points[glob_mesh.Connections[res.i][0]]);
	for (j=1;j<glob_mesh.Sizes[res.i];j++){
		d = dist(Q,&glob_mesh.Points[glob_mesh.Connections[res.i][j]]);
		if (d<dmin){
			dmin = d;
			res.j = j;
		}
	}
	free(Q);
	return res;
}

int read_poly_file(char* path,char* Name,int** Borders,int* Size0,int* Size1,point2D** Inside,int quiet){
	int i,j,k,l,l_min,size,n,sum,num,pos;
	double min,d;
	FILE* file;
	char name[512];
	char buff[512];
	int len = 0;
	sprintf(name,"%s/%s.poly",path,Name);
	file = fopen(name,"r");
	if (file==NULL){
		char Newname[512];
		strcpy(Newname,Name);
		size_t len = strlen(Newname);
		Newname[len-2] = '\0'; 						// Achtung: nur wenn mesh_index einstellig ist ! 
		sprintf(name,"%s/%s.poly",path,Newname);
		if (!quiet) printf("file %s.poly does not exist -> opening file %s.poly ... ",Name,Newname);		
		file = fopen(name,"r");
		if (file==NULL){
			printf("failed -> abort\n");
			exit(0);
		}
		else printf("\n");
	}
	readline(file,buff);
	char** line = split(buff," ",&len);
	n = atoi(line[2]);
	if (n!=2){
	if (!quiet) printf("no partition detected -> continue ...\n");
		Borders[0] = NULL;
		Borders[1] = NULL;
		*Inside = NULL;
		*Size0 = 0;
		*Size1 = 0;
		return 0;
	}
	for (j=0;j<len;j++) free(line[j]);
	free(line);
	readline(file,buff);
	line = split(buff," ",&len);
	n = atoi(line[0]);
	if (n==0){
		if (!quiet) printf("no valid segments in .poly file -> continue ...\n");
		Borders[0] = NULL;
		Borders[1] = NULL;
		*Inside = NULL;
		*Size0 = 0;
		*Size1 = 0;
		return 0;
	}
	for (j=0;j<len;j++) free(line[j]);
	free(line);
	size = 0;
	int* Labels = zero_int_list(size);
	int* Indices = zero_int_list(size);
	while(len>1){
		readline(file,buff);
		line = split(buff," ",&len);
		if (len<=1) continue;
		i = atoi(line[1]);
		l = atoi(line[3]);
		pos = in_list(Indices,size,i);
		if (pos<0){
			size++;
			Labels = (int*)realloc(Labels,size*sizeof(int));
			Indices = (int*)realloc(Indices,size*sizeof(int));
			Indices[size-1] = i;
			Labels[size-1] = l;
		}
		else Labels[pos] *= l;
		i = atoi(line[2]);
		pos = in_list(Indices,size,i);
		if (pos<0){
			size++;
			Labels = (int*)realloc(Labels,size*sizeof(int));
			Indices = (int*)realloc(Indices,size*sizeof(int));
			Indices[size-1] = i;
			Labels[size-1] = l;
		}
		else Labels[pos] *= l;
		for (j=0;j<len;j++) free(line[j]);
		free(line);
	}
	sum = 0;
	for (i=0;i<size;i++) sum += Labels[i];
	Borders[0] = zero_int_list(size-sum);
	Borders[1] = zero_int_list(sum);
	*Size0 = size-sum;
	*Size1 = sum;
	sum = 0;
	for (i=0;i<size;i++) if (Labels[i]==0){
		Borders[0][i-sum] = Indices[i];
	}
	else{
		Borders[1][sum] = Indices[i];
		sum++;
	}
	
	point2D* Q;
	sum = 0;
	for (i=0;i<(*Size0);i++){
		l = Borders[0][i];
		point2D* P = &(glob_mesh.Points[l]);
		min = 1E20;
		l_min = -1;
		for (j=0;j<glob_mesh.Sizes[l];j++){
			k = glob_mesh.Connections[l][j];
			if (is_boundary(k)>=0){
				Q = &(glob_mesh.Points[k]);
				d = dist(P,Q);
				if (d<min){
					min = d;
					l_min = k;
				}
			}
		}
		if (l_min>=0){
			sum++;
			Borders[0] = (int*)realloc(Borders[0],((*Size0)+sum)*sizeof(int));
			Borders[0][(*Size0)+sum-1] = l_min;
		}
	}
	(*Size0) += sum;
	
	free(Labels);
	free(Indices);
	if (!feof(file)){
		readline(file,buff);
		line = split(buff," ",&len);
		num = atoi(line[0]);
		for (j=0;j<len;j++) free(line[j]);
		free(line);
		*Inside = (point2D*)malloc(num*sizeof(point2D));
		for (i=0;i<num;i++){
			readline(file,buff);
			line = split(buff," ",&len);
			(*Inside)[i].x = atof(line[1]);
			(*Inside)[i].y = atof(line[2]);
			for (j=0;j<len;j++) free(line[j]);
			free(line);
		}
		if (num>1 && vec_abs(&((*Inside)[0]))!=0){
			for (i=1;i<num;i++){
				if (vec_abs(&((*Inside)[i]))==0){
					(*Inside)[i].x = (*Inside)[0].x;
					(*Inside)[i].y = (*Inside)[0].y;
					(*Inside)[0].x = 0;
					(*Inside)[0].y = 0;
					break;
				}
			}
		}
		
		fclose(file);
		return num;
	}
	else{
		fclose(file);
		return 0;
	}
}

void read_voronoi_file2D(char* path,char* Name,point2D** Edges,double* Areas,double border_width,int quiet){
	point2D* Vor_points = NULL;  //Areas mit 0 initilisieren
	point2D* A;
	point2D* B;
	point2D* Q;
	point2D slope;
	index2D node_index;
	FILE* file;
	int** Taken;
	int* na;
	int i,j,k,r,vor_len,edge_num,i1,i2;
	double x,y,l,lmax,lmean,q;
	int len = 0;
	char name[512];
	char buff[512];
	sprintf(name,"%s.v.node",Name);
	file = open_file(path,name,"r");
	readline(file,buff);
	char** line = split(buff," ",&len);
	if (len!=4){
		printf("Fehlerhaftes voronoi-node-File\n");
		exit(0);
	}
	else{
		vor_len = atoi(line[0]);
		Vor_points = (point2D*)malloc(vor_len*sizeof(point2D));
		for (j=0;j<len;j++){free(line[j]);}
		free(line);
		for (i=0;i<vor_len;i++){
			readline(file,buff);
			line = split(buff," ",&len);
			k = atoi(line[0]);
			x = atof(line[1]);
			y = atof(line[2]);
			Vor_points[k] = init_point2D(x,y);
			for (j=0;j<len;j++){free(line[j]);}
			free(line);
		}
	}
	fclose(file);
	sprintf(name,"%s.v.edge",Name);
	file = open_file(path,name,"r");
	readline(file,buff);
	line = split(buff," ",&len);
	if (len!=2){
		printf("Fehlerhaftes voronoi-edge-File\n");
		exit(0);
	}
	else{
		lmax = max_edge_len();
		lmean = border_width*mean_edge_len();
		edge_num = atoi(line[0]);
		if (!quiet){
			printf("construct voronoi cells\n");
			printf("edge number: %d\n\n",edge_num);
		}
		Taken = (int**)malloc(glob_mesh.size*sizeof(int*));
		for (i=0;i<glob_mesh.size;i++){
			Edges[i] = (point2D*)malloc(glob_mesh.Sizes[i]*sizeof(point2D));
			Taken[i] = (int*)malloc(glob_mesh.Sizes[i]*sizeof(int));
			for (j=0;j<glob_mesh.Sizes[i];j++){
				Edges[i][j].x = 0;
				Edges[i][j].y = 0;
				Taken[i][j] = 0;
			}
		}
		for (j=0;j<len;j++){free(line[j]);}
		free(line);
		for (i=0;i<edge_num;i++){
			readline(file,buff);
			line = split(buff," ",&len);
			k = atoi(line[0]);
			i1 = atoi(line[1]);
			i2 = atoi(line[2]);
			if (i2>=0){
				node_index = search_node(&Vor_points[i1],&Vor_points[i2],Taken);
				A = clone_point(&(glob_mesh.Points[glob_mesh.Connections[node_index.i][node_index.j]]));
				vec_add_mult(A,&(glob_mesh.Points[node_index.i]),-1.);
				normalize(A);
				l = dist(&Vor_points[i1],&Vor_points[i2]);
			}
			else{
				slope.x = atof(line[3]);	// ein kleiner Rand wird der Fläche hinzugefügt
				slope.y = atof(line[4]);
				normalize(&slope);
				Q = clone_point(&Vor_points[i1]);
				vec_add_mult(Q,&slope,2*lmax);
				node_index = search_node(&Vor_points[i1],Q,Taken);
				k = glob_mesh.Connections[node_index.i][node_index.j];
				A = clone_point(&(glob_mesh.Points[k]));
				B = clone_point(A);
				na = overlap(node_index.i,k);
				vec_add_mult(A,&(glob_mesh.Points[node_index.i]),-1.);
				normalize(A);
				vec_add_mult(B,&(glob_mesh.Points[node_index.i]),1.);
				vec_mult(B,0.5);
				l = dist(&Vor_points[i1],B)+lmean/sqrt(2.);
				//vec_add_mult(B,A,lmean/sqrt(2.));									// Achtung bei konvexen Gebieten !
				//vec_add_mult(B,A,lmax);
				//if (in_triangle(&glob_mesh.Points[node_index.i],A,&glob_mesh.Points[na[0]],&Vor_points[i1])<0) l = 0; 
				//l = dist(&Vor_points[i1],B);
				free(B);
				free(Q);
				free(na);
			}
			vec_mult(A,l);
			Edges[node_index.i][node_index.j].x = A->x;
			Edges[node_index.i][node_index.j].y = A->y;
			if (Taken[node_index.i][node_index.j]>0) printf("scheisen bei %d|%d\n",node_index.i,node_index.j);
			Taken[node_index.i][node_index.j] = 1;
			B = clone_point(&(glob_mesh.Points[node_index.i]));
			vec_add_mult(B,&Vor_points[i1],-1.);
			if (node_index.i<0 || node_index.i>=glob_mesh.size) printf("voll scheisen !\n");
			Areas[node_index.i] += fabs(vec_scalar(A,B))/2.;
			k = glob_mesh.Connections[node_index.i][node_index.j];
			for (j=0;j<glob_mesh.Sizes[k];j++){
				r = glob_mesh.Connections[k][j];
				if (r==node_index.i){
					Edges[k][j].x = -A->x;
					Edges[k][j].y = -A->y;
					if (Taken[k][j]>0) printf("scheisen bei %d|%d edged: %d\n",k,j,i);
					//if (k==110 && j==2) printf("blöder index: %d\n",i);
					if (k<0 || k>=glob_mesh.size) printf("voll scheisen !\n");
					Areas[k] += fabs(vec_scalar(A,B))/2.;
					Taken[k][j] = 1;
					break;
				}
			}
			free(A);
			free(B);
			for (j=0;j<len;j++){free(line[j]);}
			free(line);
		}
	}
	int counter = 0;
	for (i=0;i<glob_mesh.size;i++){
		for (j=0;j<glob_mesh.Sizes[i];j++){
			if (Taken[i][j]==0){
				printf("Fehler bei Voronoi-Konstruktion: index %d|%d nicht zugeordnet\n",i,j);
				/*k = glob_mesh.Connections[i][j];
				for (r=0;r<glob_mesh.Sizes[k];r++) if (glob_mesh.Connections[k][r]==i){
					printf("dual: %d|%d\n",k,r);
					break;
				}
				r = 0;
				for (k=0;k<glob_mesh.size;k++) r+=glob_mesh.Sizes[k];*/
				counter++;
			}
		}
		if (is_boundary(i)>=0){
			len = glob_mesh.Sizes[i]+2;
			Edges[i] = (point2D*)realloc(Edges[i],len*sizeof(point2D));
			
			k = glob_mesh.Connections[i][0];
			r = glob_mesh.Connections[i][1];
			A = clone_point(&glob_mesh.Points[k]);
			B = clone_point(&glob_mesh.Points[r]);
			vec_add_mult(A,&glob_mesh.Points[i],-1.);
			vec_add_mult(B,&glob_mesh.Points[i],-1.);
			vec_mult(A,0.5);
			Edges[i][len-2].x = -A->y;
			Edges[i][len-2].y = A->x;
			if (vec_scalar(&Edges[i][len-2],B)>0) vec_mult(&Edges[i][len-2],-1.);
			free(A);
			free(B);
			
			k = glob_mesh.Connections[i][len-3];
			r = glob_mesh.Connections[i][len-4];
			A = clone_point(&glob_mesh.Points[k]);
			B = clone_point(&glob_mesh.Points[r]);
			vec_add_mult(A,&glob_mesh.Points[i],-1.);
			vec_add_mult(B,&glob_mesh.Points[i],-1.);
			vec_mult(A,0.5);
			Edges[i][len-1].x = -A->y;
			Edges[i][len-1].y = A->x;
			if (vec_scalar(&Edges[i][len-1],B)>0) vec_mult(&Edges[i][len-1],-1.);
			free(A);
			free(B);
			
			//korrektur an der Ecke
			point2D* D1 = clone_point(&Edges[i][len-2]);
			point2D* D2 = clone_point(&Edges[i][len-1]);
			normalize(D1);
			normalize(D2);
			if (vec_scalar(D1,D2)<-1 || vec_scalar(D1,D2)>1) q = M_PI_2; 
			else q = M_PI_2-acos(vec_scalar(D1,D2))/2.;
			if (i<0 || i>=glob_mesh.size) printf("voll scheisen !\n");
			Areas[i] += lmean*(vec_abs(&Edges[i][len-1])+vec_abs(&Edges[i][len-2]))/sqrt(2.);
			if (q!=M_PI_2){
				Areas[i] += lmean*lmean/tan(q);
				vec_add_mult(&Edges[i][len-2],D1,lmean/(tan(q)*sqrt(2.)));
				vec_add_mult(&Edges[i][len-1],D2,lmean/(tan(q)*sqrt(2.)));
			}
			free(D1);
			free(D2);
		}
	}
	fclose(file);
	for (i=0;i<glob_mesh.size;i++) free(Taken[i]);
	free(Taken);
	free(Vor_points);
	if (counter>0){
		printf("%d bad connections -> abort\n",counter);
		exit(0);
	}
	/*point2D Su;
	
	for (j=0;j<glob_mesh.size;j++){
		Su.x = 0;
		Su.y = 0;
		k = glob_mesh.Sizes[j];
		if (is_boundary(j)>=0) k += 2;
		for (i=0;i<k;i++) vec_add_mult(&Su,&(Edges[j][i]),1.);
		printf("x=%f | y=%f\n",Su.x,Su.y);
	}*/
}

void insert_index(int** Connections,int n,int* Len,int ind){
	int i;
	int in = 0;
	for (i=0;i<*Len;i++){
		if (Connections[n][i]==ind){//return;
			in = 1;
			break;
		}
	}
	if (in==0){
		(*Len)++;
		Connections[n] = (int*)realloc(Connections[n],(*Len)*sizeof(int));
		Connections[n][(*Len)-1] = ind;
	}
}

double get_last_time(FILE* File,int* Mesh_size,int* Varnum,int* Step_num){
	int i,L;
	char c;
	char buffer[512];
	rewind(File);
	readline(File,buffer);
	char** List = split(buffer," ",&L);
	printf("nodenumber: %s\n",List[0]);
	printf("number of variables: %s\n",List[1]);
	printf("stepnumber: %s\n",List[2]);
	*Mesh_size = atoi(List[0]);
	*Varnum = atoi(List[1]);
	*Step_num = atoi(List[2]);
	fseek(File,-1,SEEK_END);
	long max = (long)atoi(List[0])+2;
	long counter = 0;
	do{
		c = fgetc(File);
		//printf("%s",&c);
		if (c=='\n') counter++;
		if (counter<max) fseek(File,ftell(File)-2,SEEK_SET);
	}while(counter<max);
	for (i=0;i<L;i++) free(List[i]);
	free(List);
	readline(File,buffer);
	List = split(buffer," ",&L);
	double t = atof(List[2]);
	printf("initial time: %f\n",t);
	for (i=0;i<L;i++) free(List[i]);
	free(List);
	counter = 0;
	do{
		c = fgetc(File);
		//printf("%s",&c);
		if (c=='\n') counter++;
		if (counter<max) fseek(File,ftell(File)-2,SEEK_SET);
	}while(counter<max);
	return t;
}

double** get_last_step_data(FILE* File,int mesh_size,int varnum,int stepnum){
	int i,j,k,L,I;
	char buffer[512];
	char** List;
	double** Data = (double**)malloc(2*sizeof(double*));
	Data[0] = zero_vector(mesh_size*varnum);
	Data[1] = zero_vector(mesh_size*varnum);
	for (i=0;i<=2*mesh_size;i++){
		readline(File,buffer);
		List = split(buffer," ",&L);
		if (L==3){
			for (j=0;j<L;j++) free(List[j]);
			free(List);
			continue;
		}
		if (L!=varnum){
			printf("Fehler beim Lesen des data-files: zu wenig variablen\n");
			return NULL;
		}
		k = i / mesh_size;
		I = i % mesh_size;
		if (k==1) I--;
		if (k==2){
			I = mesh_size-1;
			k = 1;
		}
		for (j=0;j<varnum;j++){
			Data[k][j*mesh_size+I] = atof(List[j]);
			free(List[j]);
		}
		free(List);
	}
	//rewind(File);
	//fprintf(File,"%d %d %d\n",mesh_size,varnum,stepnum);
	return Data;
}

void set_boundary_marker(mesh2D* Mesh){
	int i,j,k,bound;
	for (i=0;i<Mesh->size;i++){
		if (Mesh->Is_boundary[i]>=0){
			bound = 0;
			for (j=0;j<Mesh->Sizes[i];j++){
				k = Mesh->Connections[i][j];
				if (Mesh->Is_boundary[k]>=0){
					bound = 1;
					break;
				}
			}
			if (bound==1){Mesh->Is_boundary[i] = k;}
			else{printf("Gitterfehler bei Index %d am Rand\n",i);}
		}
	}
}

void read_element_file2D(char* path,char* Name,mesh2D* Mesh,index3D** Elements,
 int* tri_num,int quiet){
	int i,j,k,i1,i2,i3;
	if (Elements!=NULL) *Elements = NULL;
	int len = 0;
	char buff[256] = "";
	char name[256] = "";
	strcpy(name,Name);
	strcat(name,".ele");
	FILE* file = open_file(path,name,"r");
	readline(file,buff);
	char** line = split(buff," ",&len);
	if (len!=3){printf("Fehlerhaftes node-File\n");}
	else{
		*tri_num = atoi(line[0]);
		if (Elements!=NULL) (*Elements) = (index3D*)realloc(*Elements,(*tri_num)*sizeof(index3D));
		Mesh->Connections = (int**) malloc(Mesh->size*sizeof(int*));
		Mesh->Sizes = (int*) malloc(Mesh->size*sizeof(int));
		Mesh->Overlap_table = NULL;
		for (i=0;i<Mesh->size;i++){
			Mesh->Connections[i] = (int*)malloc(sizeof(int));
			Mesh->Sizes[i] = 0;
		}
		if (!quiet) printf("%s polygons of %s edges\n\n",line[0],line[1]);
		for (j=0;j<len;j++){free(line[j]);}
		free(line);
		for (i=0;i<(*tri_num);i++){
			readline(file,buff);
			line = split(buff," ",&len);
			k = atoi(line[0]);
			i1 = atoi(line[1]);
			i2 = atoi(line[2]);
			i3 = atoi(line[3]);
			if (Elements!=NULL){
				(*Elements)[i].i = i1;
				(*Elements)[i].j = i2;
				(*Elements)[i].k = i3;
			}
			insert_index(Mesh->Connections,i1,&(Mesh->Sizes[i1]),i2);
			insert_index(Mesh->Connections,i1,&(Mesh->Sizes[i1]),i3);
			insert_index(Mesh->Connections,i2,&(Mesh->Sizes[i2]),i1);
			insert_index(Mesh->Connections,i2,&(Mesh->Sizes[i2]),i3);
			insert_index(Mesh->Connections,i3,&(Mesh->Sizes[i3]),i1);
			insert_index(Mesh->Connections,i3,&(Mesh->Sizes[i3]),i2);
			for (j=0;j<len;j++){free(line[j]);}
			free(line);
		}
		set_boundary_marker(Mesh);
	}
	fclose(file);
}

char** read_param_strings(char* path,char* name,int* Len){
	int i,fin;
	FILE* file;
	int len = 0;
	int size = 0;
	char** line;
	char buff[256] = "";
	//char name[256] = "";
	char** params = NULL;
	*Len = 0;
	file = open_file(path,name,"r");
	if (file==NULL) file = Open_file(name,"r");
	readline(file,buff);
	line = split(buff," ",&len);
	if (len==2 && strcmp(line[0],"chemical")==0 && strcmp(line[1],"parameters:")==0){
		for (i=0;i<len;i++){
			free(line[i]);
		}
		free(line);
		fin = 0;
		size = 0;
		do {
			readline(file,buff);
			line = split(buff," ",&len);
			//if (len==2 && strcmp(line[0],"solver")==0 && strcmp(line[1],"parameters")==0){fin = 1;}
			if ((len==1 && strcmp(line[0],"end")==0) || len==0){fin = 1;}
			if (len==3){
				size++;
				params = (char**)realloc(params,size*sizeof(char*));
				params[size-1] = (char*)malloc(256*sizeof(char));
				params[size-1][0] = '\0';
				strcpy(params[size-1],line[2]);
			}
			for (i=0;i<len;i++) free(line[i]);
			free(line);
		}while(fin==0);
	}
	else{
		for (i=0;i<len;i++) free(line[i]);
		free(line);
	}
	*Len = size;
	fclose(file);
	return params;
}

/*void read_mesh2D(char* path,char* Name){
	int tri_num;
	printf("Lese Gitter aus Datei %s/%s\n\n",path,Name);
	read_node_file2D(path,Name,&glob_mesh);
	read_element_file2D(path,Name,&glob_mesh,NULL,&tri_num);
	sort_knots();
	printf("Fertig\n");
}*/

void read_mesh2D(char* path,char* Name,int quiet){
	int element_number;
	char name[256] = "";	
	//char** line;
	if (!quiet) printf("Lese Gitter aus Datei %s/%s\n\n",path,Name);
	element_number = 0;
	name[0] = '\0';
	sprintf(name,"%s",Name);
	if (!quiet) printf("%s:\n",name);
	read_node_file2D(path,name,&glob_mesh,NULL,NULL,quiet);
	read_element_file2D(path,name,&glob_mesh,NULL,&element_number,quiet);
	sort_knots();
	create_look_up_table(&glob_mesh);	
	
	int len0,len1;
	int** Borders = (int**)malloc(2*sizeof(int*));
	point2D** Seeds = (point2D**)malloc(sizeof(point2D*));
	int bnum = read_poly_file(path,name,Borders,&len0,&len1,Seeds,quiet);
	if (bnum!=0){
		int i;
		int* Partition = partition2D(NULL,Borders[0],*Seeds,len0,bnum);
		if (Partition!=NULL) init_partition(Partition,NULL,bnum);
		enable_partition();
		if (!quiet){
			printf("partitioning enabled\n");
			for (i=0;i<bnum;i++) printf("seed %d: %f,%f\n",i,(*Seeds)[i].x,(*Seeds)[i].y);
		}
	}
	else{
		disable_partition();
		if (!quiet) printf("partitioning disabled\n");
	}
	
	Dual_mesh = (point2D**)malloc(glob_mesh.size*sizeof(point2D*));
	Dual_areas = zero_vector(glob_mesh.size);
	read_voronoi_file2D(path,name,Dual_mesh,Dual_areas,1.,quiet);
}

/*void seek_trivial_rows(int** Cond,int varnum,sparse_matrix* A){
	int i,j,k,l,nontrivial;
	int d = glob_mesh.size;
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		nontrivial = 0;
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (is_boundary(k)<0){
				nontrivial++;
				break;
			}
		}
		if (!nontrivial){
			for (l=0;l<varnum;l++) if (Cond[i][j]==DIRICHLET){
				for (j=0;j<glob_mesh.Sizes[i];j++){
					k = glob_mesh.Connections[i][j];
					if (Cond[k][j]==DIRICHLET)
			}
			
		}
		
	}
}*/

void free_conditions(){
	int j;
	for (j=0;j<glob_mesh.size;j++){free(Main_conditions[j]);}
	free(Main_conditions);
}

void test_row_sum(sparse_matrix* A){
	int i,j;
	double sum,s;
	for (i=0;i<A->size;i++){
		sum = 0;
		s = 0;
		for (j=0;j<A->Len[i];j++){
			sum += A->Values[i][j];
			s += fabs(A->Values[i][j]);
		}
		if (fabs(sum/s)>1E-3) printf("sum !=0: relative %f at index %d\n",fabs(sum/s),i);
	}
}

double sqr(double x){
	return x*x;
}

double sqr2(double x){
	return x*x*x*x;
}

double* FEM_noise(double beta,double factor,int cut,int** Condi,int var_index){
	int i,k;
	double sum,x,y,s;
	int n = glob_mesh.size;
	int old_num = get_var_number();
	double* mu_x = zero_vector(cut);
	double* mu_y = zero_vector(cut);
	double* sigma = zero_vector(cut);
	double* a = zero_vector(cut);
	double* F = zero_vector(n);
	srand(time(NULL));
	for (i=0;i<cut;i++){
		mu_x[i] = (double)1.-2.*rand()/RAND_MAX;
		mu_y[i] = (double)1.-2.*rand()/RAND_MAX;
		sigma[i] = (double)0.05*rand()/RAND_MAX;
		a[i] = (double)rand()/RAND_MAX;
	}
	for (k=0;k<n;k++){
		sum = 0;
		x = (double)glob_mesh.Points[k].x;
		y = (double)glob_mesh.Points[k].y;
		for (i=0;i<cut;i++){
			sum += a[i]*exp(-((x-mu_x[i])*(x-mu_x[i])+(y-mu_y[i])*(y-mu_y[i]))/(2.*sigma[i]*sigma[i]));
			
		}
		F[k] = sum;
	}
	set_var_number(1);
	sparse_matrix* A = set_matrix_Aij_00_2D(0,0,&insert_AV);
	s = sqrt(sparse_bilinear(F,A,F));
	scalar_mult(1./s,F,n);
	
	set_var_number(old_num);
	free_sparse(A);
	free(mu_x);
	free(mu_y);
	free(sigma);
	free(a);
	return F;
}

void FEM_add_noise(double* Sol,double amplitude,int m,int var_index){
	int i,j,k,I,J,K;
	double s;
	srand(time(NULL));
	int n = glob_mesh.size;
	double* Noise = zero_vector(n);
	for (i=0;i<m;i++){
		I = rand() % n;
		s = (double)2*rand()/RAND_MAX-1.;
		Noise[I] += amplitude*s;
		for (j=0;j<glob_mesh.Sizes[I];j++){
			J = glob_mesh.Connections[I][j];
			Noise[J] = 0;
			for (k=0;k<glob_mesh.Sizes[J];k++){
				K = glob_mesh.Connections[J][k];
				Noise[J] += Noise[K];
			}
			Noise[J] /= (double)glob_mesh.Sizes[J];
		}
	}
	vector_shift(Noise,n,-get_mean(Noise,n));
	//vector_shift(Noise,n,0.01);							// Achtung: manueller shift !
	for (i=0;i<n;i++) Sol[i+n*var_index] += Noise[i];
}

void set_solution_to_zero(double* Solution,int i_var,double (*F)(double x,double y)){
	int i;
	double x,y;
	int n_mesh = glob_mesh.size;
	for (i=0;i<n_mesh;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		if ((*F)(x,y)<0) Solution[n_mesh*i_var+i] = 0;
	}
}

void Runge_Kutta4_step(double* X,int size,int var_num,double t,double dt,
 double (*F)(double* Y,int mesh_index,double t,int var_index)){
	int i,j;
	double s;
	double ds = dt/6.;
	int n_mesh = size / var_num;
	double* Y = zero_vector(var_num);
	double* K0 = zero_vector(size);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<var_num;j++) Y[j] = X[i+n_mesh*j];
		for (j=0;j<var_num;j++) K0[i+n_mesh*j] = (*F)(Y,i,t,j);
	}
	double* X1 = clone_vector(X,size);
	vector_add(X1,K0,dt/2.,size);
	
	s = t+dt/2.;
	double* K1 = zero_vector(size);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<var_num;j++) Y[j] = X1[i+n_mesh*j];
		for (j=0;j<var_num;j++) K1[i+n_mesh*j] = (*F)(Y,i,s,j);
	}
	double* X2 = clone_vector(X,size);
	vector_add(X2,K1,dt/2.,size);
	
	s = t+dt/2.;
	double* K2 = zero_vector(size);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<var_num;j++) Y[j] = X2[i+n_mesh*j];
		for (j=0;j<var_num;j++) K2[i+n_mesh*j] = (*F)(Y,i,s,j);
	}
	double* X3 = clone_vector(X,size);
	vector_add(X3,K2,dt,size);
	
	s = t+dt;
	double* K3 = zero_vector(size);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<var_num;j++) Y[j] = X3[i+n_mesh*j];
		for (j=0;j<var_num;j++) K3[i+n_mesh*j] = (*F)(Y,i,s,j);
	}
	
	for (i=0;i<size;i++) X[i] += (K0[i]+2.*(K1[i]+K2[i])+K3[i])*ds;
	
	free(K0);
	free(K1);
	free(K2);
	free(K3);
	free(X1);
	free(X2);
	free(X3);
	free(Y);
}

void Euler_step(double* X,int size,int var_num,double t,double dt,
 double (*F)(double* Y,int mesh_index,double t,int var_index)){
	int i,j;
	double s;	
	int n_mesh = size / var_num;
	double* Y = zero_vector(var_num);
	for (i=0;i<n_mesh;i++){
		for (j=0;j<var_num;j++) Y[j] = X[i+n_mesh*j];
		for (j=0;j<var_num;j++) X[i+n_mesh*j] += (*F)(Y,i,t,j)*dt;
	}
	free(Y);
}

void Runge_Kutta4(double* X,int size,int var_num,double t,double dt,int steps,
 double (*F)(double* Y,int mesh_index,double t,int var_index)){
	int i;
	double s;
	double ds = (double)dt/steps;
	for (i=0;i<steps;i++){
		s = (double)t+i*ds;
		Runge_Kutta4_step(X,size,var_num,s,ds,F);
	}
}

void memtest(){
	int N = 1;
	set_var_number2D(N);
	sparse_matrix* A = set_matrix_Aij_00_2D(0,0,&insert_AV);
	sparse_matrix* B = set_matrix_Aij_11_2D(0,0,&insert_A_bbV);
	sparse_add(A,B,-0.1);
	free_sparse(A);
	free_sparse(B);
}

void run_FEM_simulation(char* name,char* output_dir,char* mesh_name,char* mesh_dir,int steps,
int interval,double time_length,char* initial_file_name){
	char Mesh_dir[512] = "";
	char* output_name = "data";
	//init_log_file();
	smoothing_iteration_number = 1;
	
	read_mesh2D(mesh_dir,mesh_name,0);
	sprintf(Mesh_dir,"%s/%s",mesh_dir,name);
	//memtest();
	//exit(0);
	
	//if (mkdir(Mesh_Dir,0777); reload = 0; else reload = 1;
	FILE* output_file = init_file(output_dir,output_name,var_number); 
	main_solve(steps,time_length,output_file,interval,initial_file_name);
	free_mesh();
	//disable_log_file();
}
