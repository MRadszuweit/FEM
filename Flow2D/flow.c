#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <sys/stat.h>
#include "umfpack.h"
#include "linear_algebra.h"
#include "redblack.h"
#include "geometry2D.h"
#include "FEM2D.h"
#include "FEM.h"

#define CHATTY 0
#define QUIET 1

typedef struct CSR{
	int rows;					
	int* row_start;				
	int* indices;					
	double* elements;		
}CSR_matrix;

typedef struct CSC{
	int cols;	
	int* col_start;			
	int* indices;		
	double* elements;	
}CSC_matrix;

typedef struct COO_ORDERED{
	rbNode* root;
	int stride;
	int size;
}COO_ordered;

typedef struct CONTRACTION_RULE{
	int rank;
	void* func;
}contraction_rule;

typedef struct RANKN_ELEMENT{
	int* coordinates;
	int rank;
	double val;
}rankn_element;

typedef struct MESH_DATA_2D{
	mesh2D nodes;
	index3D* elements;
	index2D* edges;
	int* triangeToEdgeMap;
	int el_number;
	int edge_number;
}mesh_data_2D;

typedef enum TENSOR_TYPE{SCALAR,VECTOR,TENSOR2}tensor_type;

static double UMFPACK_pattern_info[UMFPACK_INFO];	
static double UMFPACK_factor_info[UMFPACK_INFO];	
static double UMFPACK_solve_info[UMFPACK_INFO];	

static mesh2D base_mesh;
static element_collection base_elements;
static int attribute_num = 0;
static double* Attributes = NULL;

extern mesh2D glob_mesh;
extern element_collection elements;


static int cmp_rankn_element(void* X1,void* X2){
	int i1,i2;
	rankn_element* e1 = (rankn_element*)X1;
	rankn_element* e2 = (rankn_element*)X2;
	
	int ptr = 0;
	int size = e1->rank;
	while(ptr<size){
		i1 = e1->coordinates[ptr];
		i2 = e2->coordinates[ptr];
		if (i1>i2) return 1; else if (i1<i2) return -1;
		ptr++;
	}
	return 0;
}

// i<=j 
static int cmp_rank2_symmetric(void* X1,void* X2){
	rankn_element* e1 = (rankn_element*)X1;
	rankn_element* e2 = (rankn_element*)X2;
	
	int i1 = e1->coordinates[0];
	int j1 = e1->coordinates[1];
	if (i1>j1){
		i1 = e1->coordinates[1];
		j1 = e1->coordinates[0];
	}
	int i2 = e2->coordinates[0];
	int j2 = e2->coordinates[1];
	if (i2>j2){
		i2 = e2->coordinates[1];
		j2 = e2->coordinates[0];
	}
	
	if (i1==i2){
		if (j1==j2) return 0; else return (j1>j2) ? 1 : -1;
	}
	else return (i1>i2) ? 1 : -1;
}

static void free_rankn_element(void* X){
	rankn_element* a = (rankn_element*)X;
	free(a->coordinates);
	free(a);
}

static void format_rankn_element(char* S,void* X){
	int i;
	char Buffer[16];
	
	rankn_element* ele = (rankn_element*)X;
	S[0] = '\0';
	for (i=0;i<ele->rank;i++){
		sprintf(Buffer,"%d,",ele->coordinates[i]);
		strcat(S,Buffer);
	}
	sprintf(Buffer,"\b : %f\n",ele->val);
	strcat(S,Buffer);	
}

static int cmp_rank2_element_col(void* X1,void* X2){
	int i1,i2;
	rankn_element* e1 = (rankn_element*)X1;
	rankn_element* e2 = (rankn_element*)X2;
	
	if (e1->coordinates[1]==e2->coordinates[1]){
		if (e1->coordinates[0]==e2->coordinates[0]) return 0; else return (e1->coordinates[0]>e2->coordinates[0]) ? 1 : -1;
	}
	else return (e1->coordinates[1]>e2->coordinates[1]) ? 1 : -1;
}

static int cmp_index2D(void* X1,void* X2){
	index3D* ind1 = (index3D*)X1;
	index3D* ind2 = (index3D*)X2;
	
	if (ind1->i==ind2->i){
		if (ind1->j==ind2->j) return 0; else return (ind1->j>ind2->j) ? 1 : -1;
	}
	else return (ind1->i>ind2->i) ? 1: -1;
}

static void free_index2D(void* X){
	index3D* ind = (index3D*)X;
	free(ind);
}

void print_COO_matrix(COO_ordered* A){
	RBTree_set_format(&format_rankn_element);
	RBTree_set_compare(&cmp_rankn_element);
	if (A!=NULL){
		FILE* file = fopen("/Home/damage/radszuwe/Daten/coo_matrix","w");
		if (file!=NULL){
			RBTorderedPrint(A->root,file);
			fclose(file);
		}
	}
}

COO_ordered* init_COO(int stride){
	COO_ordered* res = (COO_ordered*)malloc(sizeof(COO_ordered));
	res->root = NULL;
	res->size = 0;
	res->stride = stride;
	return res;
}

double* COO_rank2_times_vector(COO_ordered* A,int rows,int cols,double* X){
	int row,col;
	
	if (A==NULL) return NULL;
	
	if (A->size>0 && ((rankn_element*)A->root->data)->rank!=2){
		printf("Error in function %s: matrix is not of dimension two -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	double* res = zero_vector(rows);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			row = ele->coordinates[0];
			col = ele->coordinates[1];
			if (row<rows && col<cols) res[row] += ele->val*X[col];
		}
		node = RBTsuccessor(node);
	}
	
	return res;	
}

rankn_element* init_rankn_element(int rank){
	rankn_element* res = (rankn_element*)malloc(sizeof(rankn_element));
	res->coordinates = (int*)malloc(rank*sizeof(int));
	res->rank = rank;
	return res;
}

void COO_rank2_reset_row(COO_ordered* A,int row){
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	rankn_element* ele = init_rankn_element(2);
	ele->coordinates[0] = row;
	ele->coordinates[1] = 0;	
	
	rbNode* node = RBTnearestLargerOrEqualNode(A->root,ele);
	int current_row = (node!=NULL) ? ((rankn_element*)node->data)->coordinates[0] : row+1;

	while(current_row==row){
		RBTdeleteNode(&(A->root),node);
		node = RBTnearestLargerOrEqualNode(A->root,ele);
		current_row = (node!=NULL) ? ((rankn_element*)node->data)->coordinates[0] : row+1;
	}		
	free(ele);
}

rankn_element* copy_rankn_element(rankn_element* src){
	rankn_element* dst = (rankn_element*)malloc(sizeof(rankn_element));
	memcpy(dst,src,sizeof(rankn_element));
	dst->coordinates = (int*)malloc(dst->rank*sizeof(int));
	memcpy(dst->coordinates,src->coordinates,dst->rank*sizeof(int));	
	return dst;
}

static void insert_rank2(COO_ordered* B,double val,int i,int j){
	rankn_element* ele = init_rankn_element(2);		
	ele->coordinates[0] = i;
	ele->coordinates[1] = j;
	ele->val = val;
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	rbNode* node = RBTinsertElement(&(B->root),ele);
	if (node!=NULL){
		((rankn_element*)node->data)->val += val;	
		free_rankn_element(ele);
	}		
}

static void insert_rankn(COO_ordered* B,double val,int* coordinates,int rank){
	int i;
	rankn_element* ele = init_rankn_element(rank);		
	memcpy(ele->coordinates,coordinates,rank*sizeof(int));
	ele->val = val;
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	rbNode* node = RBTinsertElement(&(B->root),ele);
	if (node!=NULL){
		((rankn_element*)node->data)->val += val;	
		free_rankn_element(ele);
	}
}

static void COO_index_offset(COO_ordered* A,int* offset){
	int i;
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			for (i=0;i<ele->rank;i++) ele->coordinates[i] += offset[i];
		}
		node = RBTsuccessor(node);
	}
}

static void COO_scale(COO_ordered* A,double factor){
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL) ele->val *= factor;
		node = RBTsuccessor(node);
	}
}

void COO_rank2_set_Dirichlet(COO_ordered* A,int i){
	COO_rank2_reset_row(A,i);
	insert_rank2(A,1.,i,i);
	A->size = RBTnodeCount(A->root);
}

void COO_rank2_set_Dirichlet_symmetric(COO_ordered* A,int rows,int i,COO_ordered* RightSideMatrix,double sign){
	
	int j;	
	rbNode *node,*Rnode;
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);

	COO_rank2_reset_row(A,i);
	if (RightSideMatrix!=NULL && sign!=0){
		insert_rank2(A,sign,i,i);
		insert_rank2(RightSideMatrix,sign,i,i);
	}
	
	rankn_element* ele = init_rankn_element(2);
	
	for (j=0;j<rows;j++) if (j!=i){
		ele = init_rankn_element(2);
		ele->coordinates[0] = j;
		ele->coordinates[1] = i;
		node = RBTgetNode(A->root,ele);
		if (node!=NULL){
			ele->val = -((rankn_element*)node->data)->val;
			RBTdeleteNode(&(A->root),node);
			if (RightSideMatrix!=NULL){			
				Rnode = RBTinsertElement(&(RightSideMatrix->root),ele);
				if (Rnode!=NULL){
					((rankn_element*)Rnode->data)->val += ele->val;
					free(ele);
				}
			}
			else free(ele);
		}
		else free(ele);
	}
		
	A->size = RBTnodeCount(A->root);
	if (RightSideMatrix!=NULL) RightSideMatrix->size = RBTnodeCount(RightSideMatrix->root);
}


void COO_check(COO_ordered* A){
	
	if (A==NULL){
		printf("is NULL\n");
		return;
	}
	if (A->root==NULL){
		printf("is empty\n");
		return;
	}
	
	int i;
	
	int rank = ((rankn_element*)A->root->data)->rank;
	int* Dim = (int*)calloc(rank,sizeof(int));
	
	int counter = 0;
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			for (i=0;i<ele->rank;i++) if (ele->coordinates[i]>Dim[i]) Dim[i] = ele->coordinates[i];
			counter++;
		}
		node = RBTsuccessor(node);
	}
	
	printf("rank %d\ndimensions: ",rank);
	for (i=0;i<rank;i++) printf("%d,",Dim[i]+1);
	printf("\b \n#entries: %d\n\n",counter);
	
	free(Dim);	
}

void grad_2D_Q1(matrix2D* invFT,point2D* G){
	if (G!=NULL){
		G[0].x = -1;
		G[0].y = -1;
		mat_vec_mult(invFT,&(G[0]));
		G[1].x = 1;
		G[1].y = 0;
		mat_vec_mult(invFT,&(G[1]));
		G[2].x = 0;
		G[2].y = 1;
		mat_vec_mult(invFT,&(G[2]));		
	}
}

void grad_2D_RT(matrix2D* invFT,point2D* G){
	if (G!=NULL){
		G[0].x = 2.;
		G[0].y = 2.;
		mat_vec_mult(invFT,&(G[0]));
		G[1].x = -2.;
		G[1].y = 0;
		mat_vec_mult(invFT,&(G[1]));
		G[2].x = 0;
		G[2].y = -2.;
		mat_vec_mult(invFT,&(G[2]));		
	}
}


void linear2D_Q1_0(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int RANK = 1;
	
	int i,j;
	int coo[RANK];
	double b;

	for (i=0;i<degf;i++){
		coo[0] = index_set[i];			
		b = detF/6.;
		insert_rankn(B,b,coo,RANK);
	}
}

void linear2D_Q1Q1_00(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i,j;
	int coo[RANK];
	double b;

	for (i=0;i<degf;i++){
		coo[0] = index_set[i];		
		for (j=0;j<degf;j++){	
			coo[1] = index_set[j];	
			b = (i==j) ? detF/12. : detF/24.;
			insert_rankn(B,b,coo,RANK);
		}
	}
}

void linear2D_Q1Q1_a0(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,j,k;
	int coo[RANK];
	double b;
	double gi[DIM];
	
	b = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];	
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];		
			for (k=0;k<DIM;k++){
				coo[2] = k;	
				insert_rankn(B,gi[k]*b,coo,RANK);
			}			
		}
	}
}

void linear2D_Q1Q1_0a(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,j,k;
	int coo[RANK];
	double b;
	double gj[DIM];
	
	b = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];	
		for (j=0;j<degf;j++){			
			coo[1] = index_set[j];		
			gj[0] = gradF[j].x;
			gj[1] = gradF[j].y;
			for (k=0;k<DIM;k++){
				coo[2] = k;	
				insert_rankn(B,gj[k]*b,coo,RANK);
			}			
		}
	}
}


/*void linear2D_Q1Q1_ab(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 4;
	
	int i,j,k,l;
	int coo[RANK];
	double gi[DIM],gj[DIM];
	double b;
	
	b = detF/2.;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];	
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];			
			gj[0] = gradF[j].x;
			gj[1] = gradF[j].y;			
			for (k=0;k<DIM;k++){
				coo[2] = k;
				for (l=0;l<DIM;l++){
					coo[3] = l;
					insert_rankn(B,gi[k]*gj[l]*b,coo,RANK);	
				}
			}								
		}
	}
}*/

void linear2D_Q1Q1_aa(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i,j;
	int coo[RANK];
	double b,g;
	
	b = detF/2.;
	for (i=0;i<degf;i++){	
		coo[0] = index_set[i];	
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];			
			g = vec_scalar(&(gradF[i]),&(gradF[j]));
			insert_rankn(B,g*b,coo,RANK);			
		}
	}
}

void linear2D_X3X3_aa(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int RANK = 2;
	
	int coo[RANK];
	
	linear2D_Q1Q1_aa(B,index_set,degf,detF,gradF);
	
	coo[0] = el_index;
	coo[1] = el_index;
	insert_rankn(B,detF/90.,coo,RANK);			
}

void linear2D_Q1X3_0a(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,k;
	int coo[RANK];
	double b;
	double gj[DIM];
	
	linear2D_Q1Q1_0a(B,index_set,degf,detF,gradF);
	
	b = detF/120.;
	coo[1] = el_index;	
	for (i=0;i<degf;i++){
		gj[0] = -gradF[i].x;
		gj[1] = -gradF[i].y;	
		coo[0] = index_set[i];
		for (k=0;k<DIM;k++){
			coo[2] = k;	
			insert_rankn(B,gj[k]*b,coo,RANK);
		}		
	}
}

void linear2D_X3Q1_a0(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int j,k;
	int coo[RANK];
	double b;
	double gi[DIM];
	
	linear2D_Q1Q1_a0(B,index_set,degf,detF,gradF);
	
	b = detF/120.;
	coo[0] = el_index;	
	for (j=0;j<degf;j++){
		gi[0] = -gradF[j].x;
		gi[1] = -gradF[j].y;	
		coo[1] = index_set[j];
		for (k=0;k<DIM;k++){
			coo[2] = k;	
			insert_rankn(B,gi[k]*b,coo,RANK);
		}		
	}
}

void linear2D_Q1X3_a0(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,k;
	int coo[RANK];
	double b;
	double gi[DIM];
	
	linear2D_Q1Q1_a0(B,index_set,degf,detF,gradF);
	
	b = detF/120.;
	coo[1] = el_index;	
	for (i=0;i<degf;i++){
		gi[0] = -gradF[i].x;
		gi[1] = -gradF[i].y;	
		coo[0] = index_set[i];
		for (k=0;k<DIM;k++){
			coo[2] = k;	
			insert_rankn(B,gi[k]*b,coo,RANK);
		}		
	}
}


void linear2D_Q1X3_abb(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,k;
	int coo[RANK];
	double gi[DIM];
	
	matrix2D G  = {.xx=-1./3.,.xy=-1./6.,.yx=-1./6.,.yy=-1./3.};	
	matrix2D* IF = get_transpose_2D(IFT);
	mat_mat_mult(IFT,&G);
	mat_mat_mult(IF,&G);
	double b = trace_2D(&G);
	free(IF);
	
	coo[1] = el_index;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;	
		for (k=0;k<DIM;k++){
			coo[2] = k;
			insert_rankn(B,gi[k]*b,coo,RANK);
		}
	}	
}

void linear2D_X3X3_00(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i;
	int coo[RANK];
	double b;
	
	linear2D_Q1Q1_00(B,index_set,degf,detF,gradF);
	
	b = detF/360.;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];		
		coo[1] = el_index;	
		insert_rankn(B,b,coo,RANK);
		
		coo[1] = index_set[i];		
		coo[0] = el_index;	
		insert_rankn(B,b,coo,RANK);
	}
	
	b = detF/5040.;
	coo[0] = el_index;		
	coo[1] = el_index;	
	insert_rankn(B,b,coo,RANK);
}

/*void linear2D_X3Q1_00(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i;
	int coo[RANK];
	double b;
	
	linear2D_Q1Q1_00(B,index_set,degf,detF,gradF);
	
	b = detF/360.;
	for (i=0;i<degf;i++){
		coo[0] = el_index;	
		coo[1] = index_set[i];				
		insert_rankn(B,b,coo,RANK);
	}
}*/

void linear2D_X3X1_00(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i;
	int coo[RANK];
	double b;
	
	linear2D_Q1Q1_00(B,index_set,degf,detF,gradF);
	
	b = detF/18.;
	for (i=0;i<degf;i++){		
		coo[0] = index_set[i];		
		coo[1] = el_index;			
		insert_rankn(B,b,coo,RANK);
	}
	
	b = detF/360.;
	for (i=0;i<degf;i++){		
		coo[0] = el_index;					
		coo[1] = index_set[i];		
		insert_rankn(B,b,coo,RANK);
	}
	
	b = detF*13./3240.;
	coo[0] = el_index;					
	coo[1] = el_index;	
	insert_rankn(B,b,coo,RANK);
}

void linear2D_Q1X1_a0(COO_ordered* B,int el_index,int* index_set,int degf,matrix2D* IFT,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 3;
	
	int i,k;
	int coo[RANK];
	double b;
	double gi[DIM];
	
	linear2D_Q1Q1_a0(B,index_set,degf,detF,gradF);
	
	/*b = detF/6.;
	for (i=0;i<degf;i++){	
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		coo[0] = index_set[i];		
		coo[1] = el_index;			
		for (k=0;k<DIM;k++){
			coo[2] = k;
			insert_rankn(B,gi[k]*b,coo,RANK);
		}
	}*/
}


/*void linear2D_Q1Q1Q1_000(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){	
	const int RANK = 3;
	
	int i,j,k;
	int coo[RANK];
	double b;
	
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];	
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];	
			for (k=0;k<degf;k++){
				coo[2] = index_set[k];	
					
				if (i==j && i==k){
					b = detF/20.;
				}
				else if (i==j || i==k || j==k){
					b = detF/60.;
				}
				else b = detF/120.;
								
				insert_rankn(B,b,coo,RANK);					
			}
		}
	}	
}

void linear2D_Q1Q1Q1_100(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 4;
	
	int i,j,k,a;
	int coo[RANK];
	double b;
	double gi[DIM];
	
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];	
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];	
			for (k=0;k<degf;k++){
				coo[2] = index_set[k];	
				b = (j==k) ? detF/12. : detF/24.;
				for (a=0;a<DIM;a++){				
					coo[3] = a;
					insert_rankn(B,gi[a]*b,coo,RANK);
				}			
			}
		}
	}
}

void linear2D_Q1Q1Q1_110(COO_ordered* B,int* index_set,int degf,double detF,point2D* gradF){
	const int DIM = 2;
	const int RANK = 5;
	
	int i,j,k,a,b;
	int coo[RANK];
	double f;
	double gi[DIM],gj[DIM];
	
	f = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = index_set[i];		
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		for (j=0;j<degf;j++){
			coo[1] = index_set[j];	
			gj[0] = gradF[j].x;
			gj[1] = gradF[j].y;
			for (k=0;k<degf;k++){
				coo[2] = index_set[k];					
				for (a=0;a<DIM;a++){
					coo[3] = a;
					for (b=0;b<DIM;b++){
						coo[4] = b;
						insert_rankn(B,gi[a]*gj[b]*f,coo,RANK);
					}
				}
			}
		}
	}
}*/

void linear2D_Q0_0(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 1;
	
	int coo[RANK];
	
	double b = detF/2.;
	coo[0] = el_index;
	insert_rankn(B,b,coo,RANK);		
}

void linear2D_Q0Q0_00(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int coo[RANK];
	
	double b = detF/2.;
	coo[0] = el_index;
	coo[1] = el_index;
	insert_rankn(B,b,coo,RANK);		
}

void linear2D_RT1RT1_00(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i,j;
	int coo[RANK];
	
	double b = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = edge_set[i];				
		coo[1] = edge_set[i];	
		insert_rankn(B,b,coo,RANK);		
	}
}

void linear2D_RT1RT1_aa(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i,j;
	int coo[RANK];
	double b,g;
	
	b = detF/2.;
	for (i=0;i<degf;i++){	
		coo[0] = edge_set[i];	
		for (j=0;j<degf;j++){
			coo[1] = edge_set[j];			
			g = vec_scalar(&(gradF[i]),&(gradF[j]));
			insert_rankn(B,g*b,coo,RANK);			
		}
	}
}

void linear2D_RT1Q0_a0(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 3;
	const int DIM = 2;
	
	int i,j,a;
	int coo[RANK];
	double gi[DIM];
	
	double b = detF/2.;
	coo[1] = el_index;			
	for (i=0;i<degf;i++){	
		coo[0] = edge_set[i];	
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		
		for (a=0;a<DIM;a++){
			coo[2] = a;			
			insert_rankn(B,gi[a]*b,coo,RANK);		
		}		
	}
}

void linear2D_RT1Q1_00(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 2;
	
	int i,j;
	int coo[RANK];
	
	double b = detF/12.;
	
	for (i=0;i<degf;i++){	
		coo[0] = edge_set[i];	
		for (j=0;j<degf;j++){
			coo[1] = vertex_set[j];
			if (i!=j) insert_rankn(B,b,coo,RANK);		
		}
	}			
}

void linear2D_RT1RT1RT1_a00(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 4;
	const int DIM = 2;
	
	int i,j,k,a;
	int coo[RANK];
	double gi[DIM];
	
	double b = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = edge_set[i];		
		gi[0] = gradF[i].x;
		gi[1] = gradF[i].y;
		for (j=0;j<degf;j++) if (i==j){
			coo[1] = edge_set[j];		
			for(k=0;k<degf;k++){				
				coo[2] = edge_set[k];	
				for(a=0;a<DIM;a++){
					coo[3] = a;
					insert_rankn(B,b*gi[a],coo,RANK);		
				}
			}
		}
	}
}

/*void linear2D_RT1RT1RT1_00a(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int degf,double detF,point2D* gradF){
	const int RANK = 4;
	const int DIM = 2;
	
	int i,j,k,a;
	int coo[RANK];
	double gk[DIM];
	
	double b = detF/6.;
	for (i=0;i<degf;i++){
		coo[0] = edge_set[i];		
		for (j=0;j<degf;j++) if (i==j){
			coo[1] = edge_set[j];		
			for(k=0;k<degf;k++){
				gk[0] = gradF[k].x;
				gk[1] = gradF[k].y;
				coo[2] = edge_set[k];	
				for(a=0;a<DIM;a++){
					coo[3] = a;
					insert_rankn(B,b*gk[a],coo,RANK);		
				}
			}
		}
	}
}*/

COO_ordered* assemble_linear_2D(mesh_data_2D* Mesh, void (*assemble_single)(COO_ordered* B,int* index_set,int set_size,double detF,point2D* gradF)){
	
	const int DEGF = 3;
	
	int i,j,k,l,I,J,K;
	double b,detF,gx,gy;	
	point2D *Pi,*Pj,*Pk;	
	int set[DEGF];
	point2D gradF[DEGF];
	matrix2D F;
	matrix2D IFT;
	
	RBTree_set_compare(&cmp_rankn_element);	
	COO_ordered* B = init_COO(Mesh->nodes.size);
	for (l=0;l<Mesh->el_number;l++){
		set[0] = Mesh->elements[l].i;
		set[1] = Mesh->elements[l].j;
		set[2] = Mesh->elements[l].k;
		Pi = &(Mesh->nodes.Points[set[0]]);
		Pj = &(Mesh->nodes.Points[set[1]]);
		Pk = &(Mesh->nodes.Points[set[2]]);
		get_affine_info_2D(Pi,Pj,Pk,&F,&IFT,&detF);
		grad_2D_Q1(&IFT,gradF);		
		(*assemble_single)(B,set,DEGF,fabs(detF),gradF);						
	}
	B->size = RBTnodeCount(B->root);
	return B;
}

COO_ordered* assemble_linear_2D_X(mesh_data_2D* Mesh, void (*assemble_single)(COO_ordered* B,int el_index,int* index_set,int set_size,matrix2D* IFT,double detF,point2D* gradF)){
	
	const int DEGF = 3;
	
	int i,j,k,l;
	double detF;	
	point2D *Pi,*Pj,*Pk;	
	int set[DEGF];
	point2D gradF[DEGF];
	matrix2D F;
	matrix2D IFT;
	
	int n = Mesh->nodes.size;
	int m = Mesh->el_number;
	RBTree_set_compare(&cmp_rankn_element);	
	COO_ordered* B = init_COO(n+m);
	for (l=0;l<m;l++){
		set[0] = Mesh->elements[l].i;
		set[1] = Mesh->elements[l].j;
		set[2] = Mesh->elements[l].k;
		Pi = &(Mesh->nodes.Points[set[0]]);
		Pj = &(Mesh->nodes.Points[set[1]]);
		Pk = &(Mesh->nodes.Points[set[2]]);
		
		get_affine_info_2D(Pi,Pj,Pk,&F,&IFT,&detF);
		grad_2D_Q1(&IFT,gradF);			
		
		(*assemble_single)(B,n+l,set,DEGF,&IFT,fabs(detF),gradF);										
	}
	B->size = RBTnodeCount(B->root);
	return B;
}

static int isEdge(index2D ind,int i,int j){
	if ((ind.i==i && ind.j==j) || (ind.j==i && ind.i==j)) return 1; else return 0; 
}

static int hasVertex(index2D ind,int i){
	if (ind.i==i || ind.j==i) return 1; else return 0;
}

static void get_ordered_edges(mesh_data_2D* Mesh,int tri,int i,int j,int k,int* set){
	int e1 = Mesh->triangeToEdgeMap[3*tri];
	int	e2 = Mesh->triangeToEdgeMap[3*tri+1];
	int	e3 = Mesh->triangeToEdgeMap[3*tri+2];
	
	if (isEdge(Mesh->edges[e1],i,j)){
		set[2] = e1; //k
		if (hasVertex(Mesh->edges[e2],i)){
			set[1] = e2; //j
			set[0] = e3; //i
		}
		else{
			set[1] = e3; //j
			set[0] = e2; //i
		}
	}
	else if (isEdge(Mesh->edges[e1],j,k)){
		set[0] = e1;
		if (hasVertex(Mesh->edges[e2],j)){
			set[2] = e2;
			set[1] = e3;
		}
		else{
			set[2] = e3;
			set[1] = e2;
		}
	}
	else{	// k,i
		set[1] = e1;
		if (hasVertex(Mesh->edges[e2],k)){
			set[0] = e2;
			set[2] = e3;
		}
		else{
			set[0] = e3;
			set[2] = e2;
		}
	}
}

COO_ordered* assemble_linear_RT_2D(mesh_data_2D* Mesh, void (*assemble_single)(COO_ordered* B,int el_index,int* edge_set,int* vertex_set,int set_size,double detF,point2D* gradF)){
	
	const int DEGF = 3;
	
	int i,j,k,l,e1,e2,e3;
	double detF;
	point2D *Pi,*Pj,*Pk;	
	int Edge[DEGF],Vert[DEGF];
	point2D gradF[DEGF];
	matrix2D F;
	matrix2D IFT;

	int n = Mesh->edge_number;
	int m = Mesh->el_number;
	
	RBTree_set_compare(&cmp_rankn_element);	
	COO_ordered* B = init_COO(n);
	for (l=0;l<m;l++){
		i = Mesh->elements[l].i;
		j = Mesh->elements[l].j;
		k = Mesh->elements[l].k;		
		Pi = &(Mesh->nodes.Points[i]);
		Pj = &(Mesh->nodes.Points[j]);
		Pk = &(Mesh->nodes.Points[k]);
		get_ordered_edges(Mesh,l,i,j,k,Edge);
		
		get_affine_info_2D(Pi,Pj,Pk,&F,&IFT,&detF);
		grad_2D_RT(&IFT,gradF);			
		
		Vert[0] = i;
		Vert[1] = j;
		Vert[2] = k;
		(*assemble_single)(B,l,Edge,Vert,DEGF,fabs(detF),gradF);										
	}
	B->size = RBTnodeCount(B->root);
	return B;
}
	
void print_CSR(CSR_matrix* A){
	if (A!=NULL){
		int i,j,J;
		double a;
		
		FILE* file = fopen("/Home/damage/radszuwe/Daten/csr","w");
		if (file==NULL) return;
		
		for (i=0;i<A->rows;i++){
			for (j=A->row_start[i];j<A->row_start[i+1];j++){
				J = A->indices[j];
				a = A->elements[j];
				fprintf(file,"(%d,%e)\t",J,a);
			}
			fprintf(file,"\n");
		}		
		fclose(file);
	}
}


void print_CSC(CSC_matrix* A){
	if (A!=NULL){
		int i,j,J;
		double a;
		
		FILE* file = fopen("/Home/damage/radszuwe/Daten/csc","w");
		if (file==NULL) return;
		
		for (i=0;i<A->cols;i++){
			for (j=A->col_start[i];j<A->col_start[i+1];j++){
				J = A->indices[j];
				a = A->elements[j];
				fprintf(file,"(%d,%e)\t",J,a);
			}
			fprintf(file,"\n");
		}		
		fclose(file);
	}
}

static CSR_matrix* CSRinit(int rows,int init_size){
	CSR_matrix* res = (CSR_matrix*)malloc(sizeof(CSR_matrix));
	res->rows = rows;
	res->row_start = (int*)calloc(rows+1,sizeof(int));
	res->indices = (int*)malloc(init_size*sizeof(int));
	res->elements = (double*)malloc(init_size*sizeof(double));
	return res;
}

static CSC_matrix* CSCinit(int cols,int init_size){
	CSC_matrix* res = (CSC_matrix*)malloc(sizeof(CSC_matrix));
	res->cols = cols;
	res->col_start = (int*)calloc(cols+1,sizeof(int));
	res->indices = (int*)malloc(init_size*sizeof(int));
	res->elements = (double*)malloc(init_size*sizeof(double));
	return res;
}

static void CSR_free_matrix(CSR_matrix** const A){
	if (*A!=NULL){
		if ((*A)->row_start!=NULL) free((*A)->row_start);
		if ((*A)->indices!=NULL) free((*A)->indices);
		if ((*A)->elements!=NULL) free((*A)->elements);
		free(*A);
		*A = NULL;
	}
}

static void CSC_free_matrix(CSC_matrix** const A){
	if (*A!=NULL){
		if ((*A)->col_start!=NULL) free((*A)->col_start);
		if ((*A)->indices!=NULL) free((*A)->indices);
		if ((*A)->elements!=NULL) free((*A)->elements);
		free(*A);
		*A = NULL;
	}
}

CSR_matrix* CSRtranspose(CSR_matrix* A,int cols){
	int i,j;
	rankn_element* ele;
	
	RBTree_set_compare(&cmp_rankn_element);
	RBTree_set_free(&free_rankn_element);
	
	int rows = A->rows;
	int size = A->row_start[rows];
	CSR_matrix* TA = CSRinit(cols,size);
	
	rbNode* root = NULL;
	for (i=0;i<rows;i++){
		for (j=A->row_start[i];j<A->row_start[i+1];j++){
			ele = init_rankn_element(2);
			ele->coordinates[0] = A->indices[j];
			ele->coordinates[1] = i;
			ele->val = A->elements[j];
			if (RBTinsertElement(&root,ele)!=NULL) free(ele);
		}
	}
	
	int counter = 0;
	int prev_start = -1;		
	TA->row_start[0] = 0;
	rbNode* node = RBTminNode(root);
	while(node!=NULL){
		ele = (rankn_element*)node->data;
		TA->indices[counter] = ele->coordinates[1];
		TA->elements[counter] = ele->val;
		if (ele->coordinates[0]!=prev_start){						
			prev_start = ele->coordinates[0];
			TA->row_start[prev_start] = counter;
		}					
		node = RBTsuccessor(node);
		counter++;					
	}
	TA->row_start[cols] = counter;	
	if (counter!=size){
		printf("error in %s!\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	RBTfree(root);
	return TA;
}

CSC_matrix* CSRtoCSC(CSR_matrix* A,int cols){
	CSR_matrix* TA = CSRtranspose(A,cols);
	CSC_matrix* res = (CSC_matrix*)malloc(sizeof(CSC_matrix));
	res->cols = cols;
	res->col_start = TA->row_start;
	res->indices = TA->indices;
	res->elements = TA->elements;
	free(TA);
	return res;
}

CSR_matrix* CSCtoCSR(CSC_matrix* A,int rows){
	CSR_matrix* AT = (CSR_matrix*)malloc(sizeof(CSR_matrix));
	AT->rows = rows;
	AT->row_start = A->col_start;
	AT->indices = A->indices;
	AT->elements = A->elements;
	CSR_matrix* res = CSRtranspose(AT,rows);
	free(AT);
	return res;
}

static double* CSRmatrixTimesVector(CSR_matrix* A,double* X){
	int i,j,start;
	double sum;
	
	int end = 0;
	double* Y = zero_vector(A->rows);
	for (i=0;i<A->rows;i++){
		sum = 0;
		start = end;
		end = A->row_start[i+1];
		for (j=start;j<end;j++){			
			sum += A->elements[j]*X[A->indices[j]];
		}
		Y[i] = sum;
	}
	return Y;	
}

static double* CSCmatrixTimesVector(CSC_matrix* A,double* X){
	int i,j,start;
	double sum;
	
	int end = 0;
	double* Y = zero_vector(A->cols);
	for (i=0;i<A->cols;i++){
		sum = 0;
		start = end;
		end = A->col_start[i+1];
		for (j=start;j<end;j++){			
			Y[A->indices[j]] += A->elements[j]*X[i];			
		}		
	}
	return Y;	
}

static CSClinearMap(double* X,double factor,CSC_matrix* A,double* Y,int n){
	double* Z = CSCmatrixTimesVector(A,Y);
	vector_add(X,Z,factor,n);
	free(Z);
}


static double* CSRtranposeMatrixTimesVector(CSC_matrix* A,double* X){
	int i,j,start;
	double sum;
	
	int end = 0;
	double* Y = zero_vector(A->cols);
	for (i=0;i<A->cols;i++){
		sum = 0;
		start = end;
		end = A->col_start[i+1];
		for (j=start;j<end;j++){
			sum += A->elements[j]*X[A->indices[j]];
		}
		Y[i] = sum;
	}
	return Y;	
}

//X -> A^T.X
static void CSRtranposeMult(CSC_matrix* A,double* X){
	double* Y = CSRtranposeMatrixTimesVector(A,X);
	memcpy(X,Y,A->cols*sizeof(double));
	free(Y);
}

// Y -> Y+factor*A.X
static void CSRaddMatrixTimesVector(double* Y,double factor,CSR_matrix* A,double* X){
	int i,j,start;
	double sum;
	
	int end = 0;
	for (i=0;i<A->rows;i++){
		sum = 0;
		start = end;
		end = A->row_start[i+1];
		for (j=start;j<end;j++){
			sum += A->elements[j]*X[A->indices[j]];
		}
		Y[i] += factor*sum;
	}
}

static void CSRaddTransposedMatrixTimesVector(double* Y,double factor,CSC_matrix* A,double* X){
	int i,j,start;
	double sum;
	
	int end = 0;
	for (i=0;i<A->cols;i++){
		sum = 0;
		start = end;
		end = A->col_start[i+1];
		for (j=start;j<end;j++){
			sum += A->elements[j]*X[A->indices[j]];
		}
		Y[i] += factor*sum;
	}
}

void CSCmatrixAdd(CSC_matrix* A,CSC_matrix* B,int rows,double factor){
	int i,j,k,knext,lA,lB,size;
	
	int cols = A->cols;
	if (cols!=B->cols){
		printf("error in function %s: matrix sizes differ -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	int* start = (int*)malloc((cols+1)*sizeof(int));
	int* list = (int*)malloc(rows*cols*sizeof(int));
	double* eles = (double*)malloc(rows*cols*sizeof(double));
	
	size = 0;
	for (i=0;i<cols;i++){		
		start[i] = size;
		k = A->col_start[i];
		knext = A->col_start[i+1];
		for (j=B->col_start[i];j<B->col_start[i+1];j++){
			lB = B->indices[j];									
			while(k<knext && (lA=A->indices[k])<lB){
				list[size] = lA;
				eles[size] = A->elements[k++];
				size++;
			}
			list[size] = lB;
			eles[size] = factor*B->elements[j];
			if (k<knext && (lA=A->indices[k])==lB) eles[size] += A->elements[k++];
			size++;
		}
		while(k<knext){
			list[size] = A->indices[k];
			eles[size] = A->elements[k];
			size++;
			k++;
		}		
	}
	start[cols] = size;
	
	list = (int*)realloc(list,size*sizeof(int));
	eles = (double*)realloc(eles,size*sizeof(double));
	free(A->col_start);
	free(A->indices);
	free(A->elements);
	A->col_start = start;
	A->indices = list;
	A->elements = eles;
}

static CSC_scale(CSC_matrix* A,double factor){
	int i;	
	int N = A->col_start[A->cols];
	for (i=0;i<N;i++) A->elements[i] *= factor;
}

static double CSRresiduum(CSR_matrix* A,double* b,double* X){
	double* Y = CSRmatrixTimesVector(A,X);
	vector_add(Y,b,-1.,A->rows);
	double res = euklid_norm(Y,A->rows);
	free(Y);
	return res;
}

static void CSRmatrixAdd(CSR_matrix* A,CSR_matrix* B,int cols,double factor){
	int i,j,k,knext,lA,lB,size;
	
	int rows = A->rows;
	int* start = (int*)malloc((rows+1)*sizeof(int));
	int* list = (int*)malloc(rows*cols*sizeof(int));
	double* eles = (double*)malloc(rows*cols*sizeof(double));
	
	size = 0;
	for (i=0;i<rows;i++){		
		start[i] = size;
		k = A->row_start[i];
		knext = A->row_start[i+1];
		for (j=B->row_start[i];j<B->row_start[i+1];j++){
			lB = B->indices[j];									
			while(k<knext && (lA=A->indices[k])<lB){
				list[size] = lA;
				eles[size] = A->elements[k++];
				size++;
			}
			list[size] = lB;
			eles[size] = factor*B->elements[j];
			if (k<knext && (lA=A->indices[k])==lB) eles[size] += A->elements[k++];
			size++;
		}
		while(k<knext){
			list[size] = A->indices[k];
			eles[size] = A->elements[k];
			size++;
			k++;
		}		
	}
	start[rows] = size;
	
	list = (int*)realloc(list,size*sizeof(int));
	eles = (double*)realloc(eles,size*sizeof(double));
	free(A->row_start);
	free(A->indices);
	free(A->elements);
	A->row_start = start;
	A->indices = list;
	A->elements = eles;
}

static double CSRmatrixNorm(CSR_matrix* A){
	if (A!=NULL){
		int i,j,k;
		double sum;
		
		double max = 0;
		int n = A->rows;
		for (i=0;i<n;i++){
			sum = 0;
			for (j=A->row_start[i];j<A->row_start[i+1];j++){
				k = A->indices[j];
				sum += fabs(A->elements[k]);
			}
			if (sum>max) max = sum;
		}
		return max;
	}
	return 0;
}

static void CSC_expand(CSC_matrix* A,int new_col_size){
	int i;
	
	int n = A->cols;
	int N = A->col_start[n];
	A->col_start = (int*)realloc(A->col_start,(new_col_size+1)*sizeof(int));	
	A->cols = new_col_size;
	for (i=n+1;i<=new_col_size;i++) A->col_start[i] = N;
}

/*static void CSC_index_offset(CSC_matrix* A,int row_offset,int col_offset,int tot_cols){
	int i;
	
	int n = A->cols;
	int N = A->col_start[n];
	CSC_expand(A,tot_cols);
	memmove(&(A->col_start[col_offset]),A->col_start,(n+1)*sizeof(int));
	memset(A->col_start,0,col_offset*sizeof(int));
	
	for (i=0;i<N;i++) A->indices[i] += row_offset;
}*/

// computes A -> A+factor*B
static void COO_matrix_add(COO_ordered* A,COO_ordered* B,double factor){
	
	double val;
	
	RBTree_set_compare(&cmp_rankn_element);
	
	if (A==NULL || B==NULL) return;
	if (A->size!=0 && B->size!=0){		
		int a = ((rankn_element*)A->root->data)->rank;
		int b = ((rankn_element*)B->root->data)->rank;
		if (a!=b){
			printf("Warning in function %s : try to add matrices with different dimensions -> skip\n",__func__);
			return;
		}
	}
	
	rbNode* node = RBTminNode(B->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			rankn_element* new_ele = copy_rankn_element(ele);					
			val = new_ele->val*factor;
			new_ele->val = val;
			rbNode* new_node = RBTinsertElement(&(A->root),new_ele);
			if (new_node!=NULL){
				((rankn_element*)new_node->data)->val += val;	
				free_rankn_element(new_ele);
			}
			else A->size++;	
		}
		node = RBTsuccessor(node);
	}
	A->size = RBTnodeCount(A->root);	
}

double COO_matrix_sum_norm(COO_ordered* A){
	double sum = 0;
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL) sum += fabs(ele->val);
		node = RBTsuccessor(node);
	}
	return sum;
}

static void COO_free_matrix(COO_ordered** const A){
	if (*A!=NULL){
		if ((*A)->root!=NULL){
			RBTree_set_free(&free_rankn_element);
			RBTfree((*A)->root);			
		}
		free(*A);
		*A = NULL;
	}
}

contraction_rule create_contraction_rule(double (*func)(rankn_element* old,int* coo_new,double* X,int stride)){
	contraction_rule obj;
	obj.func = func;
	obj.rank = (int)func(NULL,NULL,NULL,0);
	return obj;
}

double Aij_Xj(rankn_element* old,int* coo_new,double* X,int stride){

	if (old==NULL) return 1;
	
	coo_new[0] = old->coordinates[0];
	return old->val*X[old->coordinates[1]];	
}

double Aija_Xja(rankn_element* old,int* coo_new,double* X,int stride){

	if (old==NULL) return 1;
	
	coo_new[0] = old->coordinates[0];
	return old->val*X[old->coordinates[1]+stride*old->coordinates[2]];	
}

double Bijk_Xk(rankn_element* old,int* coo_new,double* X,int stride){
	const int index = 2;
	
	if (old==NULL) return 2;
	
	int i = 0;
	while(i<old->rank){
		if (i!=index) *(coo_new++) = old->coordinates[i];
		i++;
	}
	return old->val*X[old->coordinates[index]];	
}

double Bijka_Xi(rankn_element* old,int* coo_new,double* X,int stride){
	const int mesh_ind = 0;
	
	if (old==NULL) return 3;
	
	int i = 0;
	while(i<old->rank){
		if (i!=mesh_ind) *(coo_new++) = old->coordinates[i];
		i++;
	}
	return old->val*X[old->coordinates[mesh_ind]];	
}

double Bijka_Xk(rankn_element* old,int* coo_new,double* X,int stride){
	const int mesh_ind = 2;
	//const int space_ind = 3;
	
	if (old==NULL) return 3;
	
	int i = 0;
	while(i<old->rank){
		if (i!=mesh_ind) *(coo_new++) = old->coordinates[i];
		i++;
	}
	int offset = 0;
	//int offset = (old->coordinates[space_ind]==0) ? 0 : stride;
	return old->val*X[old->coordinates[mesh_ind]+offset];	
}

double Bijka_Xja(rankn_element* old,int* coo_new,double* X,int stride){
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[1] = old->coordinates[2];
	return old->val*X[old->coordinates[1]+stride*old->coordinates[3]];
}

double Bijka_Xk_to_AiJ(rankn_element* old,int* coo_new,double* X,int stride){
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[1] = old->coordinates[1]+stride*old->coordinates[3];
	return old->val*X[old->coordinates[2]];
}

double Bijka_Xkb_to_AxiJ(rankn_element* old,int* coo_new,double* X,int stride){
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[1] = old->coordinates[1]+stride*old->coordinates[3];
	return old->val*X[old->coordinates[2]+0];
}

double Bijka_Xkb_to_AyiJ(rankn_element* old,int* coo_new,double* X,int stride){
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[1] = old->coordinates[1]+stride*old->coordinates[3];
	return old->val*X[old->coordinates[2]+stride];
}

double Bijka_Xja_to_Aik(rankn_element* old,int* coo_new,double* X,int stride){
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[2] = old->coordinates[2];
	return old->val*X[old->coordinates[1]+stride*old->coordinates[3]];
}

double Aija_to_AiJ(rankn_element* old,int* coo_new,double* X,int stride){	
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0];
	coo_new[1] = old->coordinates[1]+old->coordinates[2]*stride;
	
	return old->val;
}

double Aija_to_AIj(rankn_element* old,int* coo_new,double* X,int stride){	
	if (old==NULL) return 2;
	
	coo_new[0] = old->coordinates[0]+old->coordinates[2]*stride;
	coo_new[1] = old->coordinates[1];
	
	return old->val;
}

COO_ordered* contract_by_rule(COO_ordered* A,double* X,contraction_rule rule){
	double (*Func)(rankn_element* old,int* coo_new,double* X,int stride);
	
	if (A->root==NULL) return NULL;
	if (rule.func==NULL) return NULL;
		
	RBTree_set_compare(&cmp_rankn_element);
	Func = rule.func;
	
	COO_ordered* B = init_COO(A->stride);
	
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			rankn_element* new_ele = init_rankn_element(rule.rank);
			double val = Func(ele,new_ele->coordinates,X,A->stride);			
			new_ele->val = val;
			rbNode* new_node = RBTinsertElement(&(B->root),new_ele);
			if (new_node!=NULL){
				((rankn_element*)new_node->data)->val += val;	
				free_rankn_element(new_ele);
			}			
		}		
		node = RBTsuccessor(node);
	}
	
	B->size = RBTnodeCount(B->root);
	return B;	
}

COO_ordered* get_rank2_upwind(COO_ordered* A){
	
	RBTree_set_compare(&cmp_rankn_element);	
	
	COO_ordered* B = init_COO(A->stride);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL){
			rankn_element* new_ele = copy_rankn_element(ele);			
			if (new_ele->coordinates[0]!=new_ele->coordinates[1] && new_ele->val<0){
				new_ele->coordinates[0] = new_ele->coordinates[1];				
			}
			rbNode* new_node = RBTinsertElement(&(B->root),new_ele);
			if (new_node!=NULL){
				((rankn_element*)new_node->data)->val += new_ele->val;	
				free_rankn_element(new_ele);
			}			
		}
		node = RBTsuccessor(node);
	}
	B->size = RBTnodeCount(B->root);
	return B;	
}

COO_ordered* get_upwind_2D(COO_ordered* A,double* V){
	
	RBTree_set_compare(&cmp_rankn_element);	
	
	int j;
	double s;
	rankn_element *ele1,*ele2;
	double v[2];
	
	int p = A->stride;
	COO_ordered* B = init_COO(p);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		ele1 = (rankn_element*)node->data;
		if (ele1!=NULL){
			j = ele1->coordinates[1];
			
			v[0] = 0;
			v[1] = 0;
			v[ele1->coordinates[3]] = ele1->val;
			rankn_element* e = copy_rankn_element(ele1);			
			e->coordinates[3] = (ele1->coordinates[3]==0) ? 1 : 0;
			rbNode* second = RBTgetNode(A->root,e);
			if (second!=NULL){
				ele2 = (rankn_element*)second->data;
				v[ele2->coordinates[3]] = ele2->val;
			}
			free(e);
			
			s = v[0]*V[j]+v[1]*V[p+j];
			rankn_element* new_ele = copy_rankn_element(ele1);	
			if (s<0) new_ele->coordinates[1] = new_ele->coordinates[0];	
			
			rbNode* new_node = RBTinsertElement(&(B->root),new_ele);
			if (new_node!=NULL){
				((rankn_element*)new_node->data)->val += new_ele->val;	
				free_rankn_element(new_ele);
			}			
		}
		node = RBTsuccessor(node);
	}
	B->size = RBTnodeCount(B->root);
	return B;	
}


COO_ordered* COO_transpose(COO_ordered* A,int index1,int index2,int stride){
	int i;
	if (A->root==NULL) return NULL;
	int rank = ((rankn_element*)A->root->data)->rank;
	if (index1>=rank || index2>=rank) return NULL;
	RBTree_set_compare(&cmp_rankn_element);
	
	COO_ordered* B = init_COO(stride);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		rankn_element* new_ele = copy_rankn_element(ele);
		new_ele->coordinates[index1] = ele->coordinates[index2];
		new_ele->coordinates[index2] = ele->coordinates[index1];
		RBTinsertElement(&(B->root),new_ele);	
		node = RBTsuccessor(node);	
	}
	
	B->size = RBTnodeCount(A->root);
	return B;	
}

double* convert_COO_to_vector(COO_ordered* A,int rows){

	int i,j;

	RBTree_set_compare(&cmp_rankn_element);
	double* res = zero_vector(rows);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL && ele->rank==1){
			i = ele->coordinates[0];
			if (i<rows) res[i] = ele->val;
		}					
		node = RBTsuccessor(node);
	}
	
	return res;
}

double* convertQ0toQ1(mesh_data_2D* Mesh,double* X,int size){
	
	int i,k;
	double q;
	index3D ind;
	
	int n = Mesh->nodes.size;
	int m = Mesh->el_number;
	
	if (size % m!= 0){
		printf("Error in function %s: wrong vector size -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	int d = size/m;
	double* Res = zero_vector(d*n);
	for (k=0;k<d;k++){
		for (i=0;i<m;i++){
			ind = Mesh->elements[i];
			Res[k*n+ind.i] += X[k*m+i];
			Res[k*n+ind.j] += X[k*m+i];
			Res[k*n+ind.k] += X[k*m+i];
		}
		
	}
		
	for (i=0;i<n;i++){
		q = (double)Mesh->nodes.Sizes[i];
		if (IsBoundary(&(Mesh->nodes),i)>=0) q -= 1.;
		for (k=0;k<d;k++) Res[k*n+i] /= q;		
	}
	
	return Res;
}

double* convertQ1toQ0(mesh_data_2D* Mesh,double* X,int size){
	
	int i,k;
	double q;
	index3D ind;
	
	int n = Mesh->nodes.size;
	int m = Mesh->el_number;
	
	if (size % n!= 0){
		printf("Error in function %s: wrong vector size -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	int d = size/n;
	double* Res = zero_vector(d*m);
	for (k=0;k<d;k++){
		for (i=0;i<m;i++){
			ind = Mesh->elements[i];
			Res[k*m+i] = (X[k*n+ind.i]+X[k*n+ind.j]+X[k*n+ind.k])/3.;
		}
	}
	
	return Res;
}

double* convertQ1toRT1(mesh_data_2D* Mesh,double* X,int size){
	int j,k;
	double x;
	index2D ind;
	
	int m = Mesh->edge_number;
	int n = Mesh->nodes.size;
	
	if (size % n!= 0){
		printf("Error in function %s: wrong vector size -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	int d = size/n;
	double* Res = zero_vector(d*m);
	for (j=0;j<d;j++){
		for (k=0;k<m;k++){
			ind = Mesh->edges[k];
			Res[j*m+k] += (X[j*n+ind.i]+X[j*n+ind.j])/2.;
		}
	}
	
	return Res;
}

double* convertRT1toQ1(mesh_data_2D* Mesh,double* X,int size){
	int i,j,k,l;
	index2D ind;
	
	int m = Mesh->edge_number;
	int n = Mesh->nodes.size;
	
	if (size % m!= 0){
		printf("Error in function %s: wrong vector size -> abort\n",__func__);
		exit(EXIT_FAILURE);
	}
	
	int d = size/m;
	double* Res = zero_vector(d*n);
	for (j=0;j<d;j++){
		for (k=0;k<m;k++){
			ind = Mesh->edges[k];		
			Res[j*n+ind.i] += X[j*m+k];
			Res[j*n+ind.j] += X[j*m+k];
		}
	}
	
	for (i=0;i<n;i++){
		l = Mesh->nodes.Sizes[i];
		for (j=0;j<d;j++) Res[j*n+i] /= (double)l;
	}
	
	return Res;
}

void print_converted(mesh_data_2D* mesh,double* X){
	
	int n = mesh->nodes.size;
	int p = mesh->edge_number;
	int m = mesh->el_number;
	
	double* V = convertRT1toQ1(mesh,&(X[0]),2*p);
	double* P = convertQ0toQ1(mesh,&(X[2*p]),m);	
	double* Y = zero_vector(3*n);
	memcpy(Y,P,n*sizeof(double));
	memcpy(&(Y[n]),V,2*n*sizeof(double));
	print_vector(Y,3*n);
	
	free(V);
	free(P);
	free(Y);
}

void print_lumped_vector(mesh_data_2D* mesh,double* X){
	int i;
	
	int n = mesh->nodes.size;
	COO_ordered* A0 = assemble_linear_2D(mesh,&linear2D_Q1_0);
	double* V = convert_COO_to_vector(A0,n);
	double* Y = clone_vector(X,n);
	for (i=0;i<n;i++){
		Y[i] /= V[i];
	}
	
	print_vector(Y,n);
	
	COO_free_matrix(&A0);
	free(V);
	free(Y);
}

void shift_Q0_by_mean(mesh_data_2D* mesh,double* X){
	int i;
	
	int m = mesh->el_number;
	COO_ordered* A0 = assemble_linear_RT_2D(mesh,&linear2D_Q0_0);
	double* V = convert_COO_to_vector(A0,m);
	double a = vector_contraction(V,m);
	double mean = scalar(V,X,m)/a;
	
	for (i=0;i<m;i++) X[i] -= mean;
	
	COO_free_matrix(&A0);
	free(V);
}

void expand_vector(double** X,int oldsize,int newsize){
	*X = (double*)realloc(*X,newsize*sizeof(double));
	if (newsize>oldsize) memset(&((*X)[oldsize]),0,(newsize-oldsize)*sizeof(double));
}

// computes <f,phi> with \phi\in X3
void get_func2D_times_scalarX3(double* X,mesh_data_2D* mesh,double (*func)(double x,double y,int a)){
	
	int i;
	double x,y,fx,fy;
	index3D ind;
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int N = n+m;
	
	double* F = zero_vector(2*N);
	for (i=0;i<n;i++){
		x = mesh->nodes.Points[i].x;
		y = mesh->nodes.Points[i].y;
		F[i] = (*func)(x,y,0);
		F[N+i] = (*func)(x,y,1);
	}
	
	for (i=0;i<m;i++){
		ind.i = mesh->elements[i].i;
		ind.j = mesh->elements[i].j;
		ind.k = mesh->elements[i].k;
		
		x = (mesh->nodes.Points[ind.i].x+mesh->nodes.Points[ind.j].x+mesh->nodes.Points[ind.k].x)/3.;
		y = (mesh->nodes.Points[ind.i].y+mesh->nodes.Points[ind.j].y+mesh->nodes.Points[ind.k].y)/3.;
		fx = (F[ind.i]+F[ind.j]+F[ind.k])/3.;
		fy = (F[ind.i+N]+F[ind.j+N]+F[ind.k+N])/3.;
		F[n+i] = (*func)(x,y,0)-fx;
		F[N+n+i] = (*func)(x,y,1)-fy;
	}
	
	COO_ordered* X00 = assemble_linear_2D_X(mesh,&linear2D_X3X1_00);
	COO_ordered* X00Fx = contract_by_rule(X00,&(F[0]),create_contraction_rule(&Aij_Xj));
	COO_ordered* X00Fy = contract_by_rule(X00,&(F[N]),create_contraction_rule(&Aij_Xj));
	double* Fx = convert_COO_to_vector(X00Fx,N);
	double* Fy = convert_COO_to_vector(X00Fy,N);
	memcpy(&(X[0]),Fx,N*sizeof(double));
	memcpy(&(X[N]),Fy,N*sizeof(double));
	
	COO_free_matrix(&X00);
	COO_free_matrix(&X00Fx);
	COO_free_matrix(&X00Fy);
	free(Fx);
	free(Fy);
	free(F);
}

// computes <f,\dot\nabla phi> with \phi\in X1
void get_func2D_times_gradX1(double* X,mesh_data_2D* mesh,double (*func)(double x,double y,int a)){
	
	int i;
	double x,y,fx,fy;
	index3D ind;
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int N = n+m;
	
	double* F = zero_vector(2*N);
	for (i=0;i<n;i++){
		x = mesh->nodes.Points[i].x;
		y = mesh->nodes.Points[i].y;
		F[i] = (*func)(x,y,0);
		F[N+i] = (*func)(x,y,1);
	}
	
	for (i=0;i<m;i++){
		ind.i = mesh->elements[i].i;
		ind.j = mesh->elements[i].j;
		ind.k = mesh->elements[i].k;
		
		x = (mesh->nodes.Points[ind.i].x+mesh->nodes.Points[ind.j].x+mesh->nodes.Points[ind.k].x)/3.;
		y = (mesh->nodes.Points[ind.i].y+mesh->nodes.Points[ind.j].y+mesh->nodes.Points[ind.k].y)/3.;
		fx = (F[ind.i]+F[ind.j]+F[ind.k])/3.;
		fy = (F[ind.i+N]+F[ind.j+N]+F[ind.k+N])/3.;
		F[n+i] = (*func)(x,y,0)-fx;
		F[N+n+i] = (*func)(x,y,1)-fy;
	}
	
	COO_ordered* Xa0 = assemble_linear_2D_X(mesh,&linear2D_Q1X1_a0);
	COO_ordered* Xa0Fa = contract_by_rule(Xa0,F,create_contraction_rule(&Aija_Xja));
	double* Y = convert_COO_to_vector(Xa0Fa,n);
	memcpy(X,Y,n*sizeof(double));
	
	COO_free_matrix(&Xa0);
	COO_free_matrix(&Xa0Fa);
	free(F);
	free(Y);
}


CSR_matrix* convert_COO_to_CSR(COO_ordered* A,int rows,double tol){

	int i,j,k;

	RBTree_set_compare(&cmp_rankn_element);
	int size = A->size;	
	int counter = 0;
	int prev_start = -1;		
	CSR_matrix* TA = CSRinit(rows,size);
	rbNode* node = RBTminNode(A->root);
	while(node!=NULL){
		rankn_element* ele = (rankn_element*)node->data;
		if (ele!=NULL && ele->rank==2 && fabs(ele->val)>tol){
			i = ele->coordinates[0];
			j = ele->coordinates[1];
			TA->indices[counter] = j;
			TA->elements[counter] = ele->val;
			if (i!=prev_start){										
				for (k=prev_start+1;k<=i;k++) TA->row_start[k] = counter;
				prev_start = i;
			}
			counter++;
		}					
		node = RBTsuccessor(node);
	}
	TA->row_start[rows] = counter;	
	/*if (counter!=size){
		printf("error in %s!\n",__func__);
		exit(EXIT_FAILURE);
	}*/
	return TA;
}

CSC_matrix* convert_COO_to_CSC(COO_ordered* A,int rows,int cols,double tol){
	COO_ordered* AT = COO_transpose(A,0,1,rows);
	CSR_matrix* ATrow = convert_COO_to_CSR(AT,cols,tol);
	COO_free_matrix(&AT);
	
	CSC_matrix* Acol = (CSC_matrix*)malloc(sizeof(CSC_matrix));
	Acol->cols = cols;
	Acol->col_start = ATrow->row_start;
	Acol->indices = ATrow->indices;
	Acol->elements = ATrow->elements;
	
	free(ATrow);
	return Acol;
}

int umfpack_solver_col(CSC_matrix* A,double* X,double* b){
	const int system_type = UMFPACK_A; 														 // default: solves Ax=b
	
	double Control [UMFPACK_CONTROL];
	umfpack_di_defaults (Control) ;
	//Control[UMFPACK_PIVOT_TOLERANCE] = 1e-8;
	//Control [UMFPACK_SYM_PIVOT_TOLERANCE] = 1e-6;

	int n = A->cols;
	static void* Pattern_info = NULL;
	static void* Factor_info = NULL;
	
	int status = umfpack_di_symbolic(n,n,A->col_start,A->indices,A->elements,&Pattern_info,Control,UMFPACK_pattern_info);
	if (status!=UMFPACK_OK){
		printf("Error in UMFPACK symbolic -> abort\n");
		exit(EXIT_FAILURE);
	}
	
	status = umfpack_di_numeric(A->col_start,A->indices,A->elements,Pattern_info,&Factor_info,Control,UMFPACK_factor_info);
	if (status!=UMFPACK_OK){
		printf("Error in UMFPACK numeric -> abort\n");
		exit(EXIT_FAILURE);	
	}
	
	status = umfpack_di_solve(system_type,A->col_start,A->indices,A->elements,X,b,Factor_info,Control,UMFPACK_solve_info);
	if (status!=UMFPACK_OK){
		printf("Error in UMFPACK solve -> abort\n");
		exit(EXIT_FAILURE);	
	}	
	
	umfpack_di_free_symbolic(&Pattern_info);
	umfpack_di_free_numeric(&Factor_info);	
	
	return UMFPACK_OK;
}

int umfpack_solver(CSR_matrix* A,double* X,double* b,int n){
	CSC_matrix* Acol = CSRtoCSC(A,n);
	int status = umfpack_solver_col(Acol,X,b);
	CSC_free_matrix(&Acol);
	return status;
}

void insert_edge(rbNode** Edges,int i,int j,int tri_index){
	rbNode* node;
	
	rankn_element* edge_info = init_rankn_element(4);
	edge_info->coordinates[0] = i;
	edge_info->coordinates[1] = j;
	edge_info->coordinates[2] = -1;
	edge_info->coordinates[3] = -1;
	
	if ((node=RBTinsertElement(Edges,edge_info))==NULL){
		edge_info->coordinates[2] = tri_index;
	}
	else{
		((rankn_element*)node->data)->coordinates[3] = tri_index;
		free(edge_info);			
	}
}

void get_edge_collection(index3D* elements,int el_num,index2D** edges,int* ed_num,int** triangle_to_edge_map){
	int i,j,k,l;
	rankn_element* edge_info;
	
	RBTree_set_compare(&cmp_rank2_symmetric);
	RBTree_set_free(&free_rankn_element);
	
	rbNode* Edges = NULL;
	for (l=0;l<el_num;l++){
		i = elements[l].i;
		j = elements[l].j;
		k = elements[l].k;
		
		insert_edge(&Edges,i,j,l);
		insert_edge(&Edges,j,k,l);
		insert_edge(&Edges,k,i,l);
	}
	
	*ed_num = RBTnodeCount(Edges);
	*edges = (index2D*)malloc((*ed_num)*sizeof(index2D));	
	*triangle_to_edge_map = (int*)malloc(3*el_num*sizeof(int));
	int** ptrs = (int**)malloc(el_num*sizeof(int*));
	for (l=0;l<el_num;l++) ptrs[l] = &((*triangle_to_edge_map)[3*l]);
	
	i = 0;
	rbNode* node = RBTminNode(Edges);
	while(node!=NULL){
		edge_info = (rankn_element*)node->data;
		(*edges)[i].i = edge_info->coordinates[0];
		(*edges)[i].j = edge_info->coordinates[1];
		if ((l=edge_info->coordinates[2])>=0){
			*(ptrs[l]++) = i;			
		}
		if ((l=edge_info->coordinates[3])>=0){
			*(ptrs[l]++) = i;	
		}
		
		i++;
		node = RBTsuccessor(node);
	}
	
	free(ptrs);
	RBTfree(Edges);		
} 

void read_mesh_2D(char* Dir,char* Name,mesh_data_2D* Mesh,double** Attributes,int* attr_num,int quiet){
	if (!quiet) printf("read mesh from file %s/%s\n\n",Dir,Name);
	
	int element_number = 0;
	read_node_file2D(Dir,Name,&(Mesh->nodes),Attributes,attr_num,quiet);
	read_element_file2D(Dir,Name,&(Mesh->nodes),&(Mesh->elements),&element_number,quiet);
	Mesh->el_number = element_number;
	Sort_knots(&(Mesh->nodes));
	create_look_up_table(&(Mesh->nodes));
	get_edge_collection(Mesh->elements,element_number,&(Mesh->edges),&(Mesh->edge_number),&(Mesh->triangeToEdgeMap));
}

void check_Laplace(mesh_data_2D* mesh){
	
	const double tol = 1e-8;
	
	double func(double x,double y){
		return cos(M_PI*x)*sin(M_PI*y);
	}
	
	int i;
	index3D ind;
	double u,x,y;
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int N = m+n;
	
	COO_ordered* Id13 = assemble_linear_2D_X(mesh,&linear2D_X3X1_00);
	COO_ordered* Id33 = assemble_linear_2D_X(mesh,&linear2D_X3X3_00);
	
	double* F = zero_vector(N);
	for (i=0;i<n;i++){
		x = mesh->nodes.Points[i].x;
		y = mesh->nodes.Points[i].y;
		F[i] = func(x,y);
	}
	
	for (i=0;i<m;i++){
		ind.i = mesh->elements[i].i;
		ind.j = mesh->elements[i].j;
		ind.k = mesh->elements[i].k;		
		x = (mesh->nodes.Points[ind.i].x+mesh->nodes.Points[ind.j].x+mesh->nodes.Points[ind.k].x)/3.;
		y = (mesh->nodes.Points[ind.i].y+mesh->nodes.Points[ind.j].y+mesh->nodes.Points[ind.k].y)/3.;
		u = (F[ind.i+N]+F[ind.j+N]+F[ind.k+N])/3.;
		F[n+i] = func(x,y)-u;
	}
	
	COO_ordered* Id3 = contract_by_rule(Id13,F,create_contraction_rule(&Aij_Xj));
	double* f = convert_COO_to_vector(Id3,N);
	
	CSC_matrix* Id = convert_COO_to_CSC(Id33,N,N,tol);
	
	double* X = zero_vector(N);
	umfpack_solver_col(Id,X,f);
	print_vector(X,N);
	exit(0);
}

void init_Chorin_Uzawa(
	mesh_data_2D* mesh,
	CSC_matrix** Op_v,
	CSC_matrix** Lap_p,
	CSC_matrix** Id1,
	CSC_matrix** Id2,
	CSR_matrix** Div,
	CSR_matrix** Grad,
	double dt
	){
		
	const double eta = 1.0;
	const double tol = 1e-8;
	
	int i;
	int offset_yy[2];
	
	int n = mesh->nodes.size;
	offset_yy[0] = n;
	offset_yy[1] = n;
	
	COO_ordered* Vxx = assemble_linear_2D(mesh,&linear2D_Q1Q1_aa);	
	COO_ordered* Vyy = assemble_linear_2D(mesh,&linear2D_Q1Q1_aa);
	COO_index_offset(Vyy,offset_yy);
	COO_matrix_add(Vyy,Vxx,1.);	
	COO_scale(Vxx,-1.);
	*Lap_p = convert_COO_to_CSC(Vxx,n,n,tol); 	
	
	COO_ordered* Idxx = assemble_linear_2D(mesh,&linear2D_Q1Q1_00);
	COO_ordered* Idyy = assemble_linear_2D(mesh,&linear2D_Q1Q1_00);
	COO_index_offset(Idyy,offset_yy);	
	COO_matrix_add(Idyy,Idxx,1.);
	
	*Id1 = convert_COO_to_CSC(Idxx,n,n,tol);	
	*Id2 = convert_COO_to_CSC(Idyy,2*n,2*n,tol);	
	
	//COO_scale(Idyy,0);
	COO_matrix_add(Idyy,Vyy,dt*eta);
	for (i=0;i<n;i++){
		if (IsBoundary(&(mesh->nodes),i)>=0){
			COO_rank2_set_Dirichlet(Idyy,i);
			COO_rank2_set_Dirichlet(Idyy,i+n);			
		}
	}	
	*Op_v = convert_COO_to_CSC(Idyy,2*n,2*n,tol);
	
	COO_free_matrix(&Idxx);
	COO_free_matrix(&Idyy);
	COO_free_matrix(&Vxx);
	COO_free_matrix(&Vyy);
	
	COO_ordered* A01 = assemble_linear_2D(mesh,&linear2D_Q1Q1_0a);
	COO_ordered* div = contract_by_rule(A01,NULL,create_contraction_rule(&Aija_to_AiJ));
	*Div = convert_COO_to_CSR(div,n,tol);
	COO_free_matrix(&A01);
	COO_free_matrix(&div);

	
	COO_ordered* A10 = assemble_linear_2D(mesh,&linear2D_Q1Q1_a0);
	COO_ordered* grad = contract_by_rule(A10,NULL,create_contraction_rule(&Aija_to_AIj));
	*Grad = convert_COO_to_CSR(grad,2*n,tol);
	COO_free_matrix(&A10);
	COO_free_matrix(&grad);
}

void Chorin_Uzawa_step(
	mesh_data_2D* mesh,
	CSC_matrix* Op_v,
	CSC_matrix* Lap_p,
	CSC_matrix* Id1,
	CSC_matrix* Id2,
	CSR_matrix* Div,
	CSR_matrix* Grad,
	double* F,
	double* X,
	double dt
	){
		
	const double alpha = 0.3;
	
	int i;
	
	int n = Div->rows;
	double* V = &(X[0]);
	double* P = &(X[2*n]);
	double* Q = &(X[3*n]);
	
	double* f1 = CSRtranposeMatrixTimesVector(Id2,F);
	CSRaddMatrixTimesVector(f1,-1.,Grad,P);
	scalar_mult(dt,f1,2*n);
	CSRaddTransposedMatrixTimesVector(f1,1.,Id2,V);
	for (i=0;i<n;i++) if (IsBoundary(&(mesh->nodes),i)>=0){
		f1[i] = 0;	
		f1[i+n] = 0;	
	}
	double* W = zero_vector(2*n);
	umfpack_solver_col(Op_v,W,f1);
	free(f1);
	
	double* f2 = CSRmatrixTimesVector(Div,W);
	scalar_mult(1./dt,f2,n);
	umfpack_solver_col(Lap_p,Q,f2);
	free(f2);	
	
	double* f3 = CSRtranposeMatrixTimesVector(Id2,V);
	CSRaddMatrixTimesVector(f3,-alpha*dt,Grad,Q);
	umfpack_solver_col(Id2,V,f3);
	free(f3);
	
	vector_add(P,Q,alpha,n);
	free(W);
}

CSC_matrix* NSassemble_Q1X3_matrix(mesh_data_2D* mesh,CSR_matrix** Stab,double dt){
	const double alpha = 1.0;
	const double eta = 1.0;
	const double tol = 1e-8;
	
	int i;
	int offset_yy[2],offset_0p[2],offset_p0[2],offset_pp[2];
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int N = n+m;
	double h = get_longest_edge(&(mesh->nodes));
	
	offset_yy[0] = N; offset_yy[1] = N;
	offset_0p[0] = 0; offset_0p[1] = 2*N;
	offset_p0[0] = 2*N; offset_p0[1] = 0;
	offset_pp[0] = 2*N; offset_pp[1] = 2*N;
	
	// Stokes matrix
	COO_ordered* IdV_xx = assemble_linear_2D_X(mesh,&linear2D_X3X3_00);
	COO_ordered* Operator = assemble_linear_2D_X(mesh,&linear2D_X3X3_00);
	COO_index_offset(Operator,offset_yy);	
	COO_matrix_add(Operator,IdV_xx,1.);
	COO_free_matrix(&IdV_xx);
	
	COO_ordered* Vis_xx = assemble_linear_2D_X(mesh,&linear2D_X3X3_aa);	
	COO_ordered* Vis_yy = assemble_linear_2D_X(mesh,&linear2D_X3X3_aa);
	COO_index_offset(Vis_yy,offset_yy);
	COO_matrix_add(Operator,Vis_xx,eta*dt);
	COO_matrix_add(Operator,Vis_yy,eta*dt);
	COO_free_matrix(&Vis_xx);
	COO_free_matrix(&Vis_yy);
	
	COO_ordered* A10 = assemble_linear_2D_X(mesh,&linear2D_X3Q1_a0);
	COO_ordered* grad = contract_by_rule(A10,NULL,create_contraction_rule(&Aija_to_AIj));
	COO_index_offset(grad,offset_0p);	
	//COO_matrix_add(Operator,grad,dt);
	COO_free_matrix(&A10);
	COO_free_matrix(&grad);
	
	COO_ordered* A01 = assemble_linear_2D_X(mesh,&linear2D_Q1X3_0a);
	COO_ordered* div = contract_by_rule(A01,NULL,create_contraction_rule(&Aija_to_AiJ));
	COO_index_offset(div,offset_p0);	
	//COO_matrix_add(Operator,div,dt);
	
COO_ordered* Id_pp = assemble_linear_2D(mesh,&linear2D_Q1Q1_00);	
COO_index_offset(Id_pp,offset_pp);
COO_matrix_add(Operator,Id_pp,dt);
COO_free_matrix(&Id_pp);	
	
	COO_free_matrix(&A01);
	COO_free_matrix(&div);
	
	// pressure stabilization (BC for p ?)
	COO_ordered* stab = init_COO(N);
	
	COO_ordered* X10 = assemble_linear_2D_X(mesh,&linear2D_Q1X3_a0);	
	COO_ordered* id_part = contract_by_rule(X10,NULL,create_contraction_rule(&Aija_to_AiJ));
	//COO_index_offset(id_part,offset_p0);
	COO_matrix_add(stab,id_part,h*h*alpha/eta);
	COO_free_matrix(&X10);
	COO_free_matrix(&id_part);
	
	COO_ordered* X111 = assemble_linear_2D_X(mesh,&linear2D_Q1X3_abb);
	COO_ordered* vis_part = contract_by_rule(X111,NULL,create_contraction_rule(&Aija_to_AiJ));
	//COO_index_offset(vis_part,offset_p0);
	COO_matrix_add(stab,vis_part,-h*h*dt*alpha);
	COO_free_matrix(&X111);
	COO_free_matrix(&vis_part);
	
	COO_ordered* p_part = assemble_linear_2D(mesh,&linear2D_Q1Q1_aa);
	//COO_index_offset(p_part,offset_pp);
COO_index_offset(p_part,offset_0p);
	//COO_matrix_add(stab,p_part,h*h*dt*alpha/eta);
	COO_free_matrix(&p_part);	
	
	// set boundary BC
	for (i=0;i<n;i++){
		if (IsBoundary(&(mesh->nodes),i)>=0){
			COO_rank2_set_Dirichlet(Operator,i);
			COO_rank2_set_Dirichlet(Operator,i+N);	
		}
	}
	
	
	*Stab = convert_COO_to_CSR(stab,n,tol);
	
	// convert to CSC format
	return convert_COO_to_CSC(Operator,2*N+n,2*N+n,tol);
}

double* NSassemble_Q1X3_inhomogeneity(mesh_data_2D* mesh,double** Stab,double* X,double dt){
	const double alpha = 1.0;
	const double eta = 1.0;
	
	double f2D(double x,double y,int a){
		return (a==0) ? 0.0 : -1.0*sin(M_PI*y);
	};
	
	int i;	
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int N = n+m;
	double h = get_longest_edge(&(mesh->nodes));
	
	double* b = zero_vector(2*N+n);
	double* VX = &(b[0]);
	double* VY = &(b[N]);
	double* P = &(b[2*N]);
	double* V0X = &(X[0]);
	double* V0Y = &(X[N]);	
	
	// Stokes part
	get_func2D_times_scalarX3(b,mesh,&f2D);
	scalar_mult(dt,b,2*N+n);
	
	COO_ordered* IdV = assemble_linear_2D_X(mesh,&linear2D_X3X3_00);
	COO_ordered* Vx = contract_by_rule(IdV,V0X,create_contraction_rule(&Aij_Xj));
	COO_ordered* Vy = contract_by_rule(IdV,V0Y,create_contraction_rule(&Aij_Xj));
	double* vx = convert_COO_to_vector(Vx,N);
	double* vy = convert_COO_to_vector(Vy,N);
	vector_add(VX,vx,1.,N);
	vector_add(VY,vy,1.,N);
	COO_free_matrix(&IdV);
	COO_free_matrix(&Vx);
	COO_free_matrix(&Vy);
	free(vx);
	free(vy);
	
	// pressure stabilization
	*Stab = zero_vector(n);
	COO_ordered* X10 = assemble_linear_2D_X(mesh,&linear2D_Q1X3_a0);	
	COO_ordered* id_part = contract_by_rule(X10,V0X,create_contraction_rule(&Aija_Xja));
	double* v0 = convert_COO_to_vector(id_part,n);
	vector_add(*Stab,v0,h*h*alpha/eta,n);
	COO_free_matrix(&X10);
	COO_free_matrix(&id_part);
	free(v0);
	
	double* f = zero_vector(n);
	get_func2D_times_gradX1(f,mesh,&f2D);
	vector_add(*Stab,f,h*h*dt*alpha/eta,n);
	
	// set boundary BC
	for (i=0;i<n;i++){
		if (IsBoundary(&(mesh->nodes),i)>=0){
			b[i] = 0;
			b[N+i] = 0;			
		}
	}
	
	free(f);
	return b;
}

CSC_matrix* CSCcopy(CSC_matrix* A){
	int cols = A->cols;
	int size = A->col_start[cols];
	CSC_matrix* res = CSCinit(cols,size);
	
	memcpy(res->col_start,A->col_start,(cols+1)*sizeof(int));
	memcpy(res->indices,A->indices,size*sizeof(int));
	memcpy(res->elements,A->elements,size*sizeof(double));
	return res;
}

void print_edge_indices(mesh_data_2D* mesh){
	int i;
	
	int p = mesh->edge_number;
	printf("\n");
	for (i=0;i<p;i++) if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
		printf("%d\n",i);
	}
	printf("\n");
}

void assembleStationaryStokes2D_RT1Q0(mesh_data_2D* mesh,double* Boundval,CSC_matrix** A,double** b){
	
	const double tol = 1e-10;
	
	int i;
	double x,y;
	
	int n = mesh->nodes.size;
	int m = mesh->el_number;
	int p = mesh->edge_number;
	int offset_yY[2] = {p,n};
	int offset_yy[2] = {p,p};
	int offset_0p[2] = {0,2*p};
	int offset_pp[2] = {2*p,2*p};	
	
	double* f = zero_vector(2*n);	
	for (i=0;i<n;i++){
		x = mesh->nodes.Points[i].x;
		y = mesh->nodes.Points[i].y;
		f[i] = -10.*exp(-2.*y)*sin(4*x);
		f[n+i] = 10.;	
	}
	
	COO_ordered* F = assemble_linear_RT_2D(mesh,&linear2D_RT1Q1_00);
	COO_ordered* Fyy = assemble_linear_RT_2D(mesh,&linear2D_RT1Q1_00);
	COO_index_offset(Fyy,offset_yY);
	COO_matrix_add(F,Fyy,1.);
	*b = COO_rank2_times_vector(F,2*p,2*n,f);
	COO_free_matrix(&Fyy);
	free(f);	
	
	COO_ordered* L = assemble_linear_RT_2D(mesh,&linear2D_RT1RT1_aa);
	COO_ordered* Lyy = assemble_linear_RT_2D(mesh,&linear2D_RT1RT1_aa);
	COO_index_offset(Lyy,offset_yy);
	COO_matrix_add(L,Lyy,1.);
	COO_scale(L,-1.);
	COO_free_matrix(&Lyy);
	
	
	COO_ordered* A10 = assemble_linear_RT_2D(mesh,&linear2D_RT1Q0_a0);
	COO_ordered* P = contract_by_rule(A10,NULL,create_contraction_rule(&Aija_to_AIj));
	COO_free_matrix(&A10);
	COO_index_offset(P,offset_0p);
	COO_ordered* PT = COO_transpose(P,0,1,p);
	COO_matrix_add(L,P,1.);
	COO_matrix_add(L,PT,1.);
	
	/*COO_ordered* S = assemble_linear_RT_2D(mesh,&linear2D_Q0Q0_00);
	COO_index_offset(S,offset_pp);
	COO_matrix_add(L,S,-1e-4);
	COO_free_matrix(&S);*/
	
	COO_ordered* BoundMatrix = init_COO(p);
	for (i=0;i<p;i++){		
		if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
			COO_rank2_set_Dirichlet_symmetric(L,2*p+m,i,BoundMatrix,-1.);
			COO_rank2_set_Dirichlet_symmetric(L,2*p+m,p+i,BoundMatrix,-1.);
			(*b)[i] = 0;
			(*b)[p+i] = 0;
		}
	}
	double* bm = COO_rank2_times_vector(BoundMatrix,2*p+m,2*p+m,Boundval);
	expand_vector(b,2*p,2*p+m);
	vector_add(*b,bm,1.,2*p+m);
	
	*A = convert_COO_to_CSC(L,2*p+m,2*p+m,tol);
	
	COO_free_matrix(&L);
	COO_free_matrix(&F);
	COO_free_matrix(&P);
	COO_free_matrix(&PT);
	COO_free_matrix(&BoundMatrix);
	free(bm);	
}

double* assembleUpwindFlux2D_RT(mesh_data_2D* mesh,COO_ordered* Ba00,double* Boundval,double* V0){
	
	int i;
	
	int p = mesh->edge_number;
	int m = mesh->el_number;
	
	for (i=0;i<p;i++) if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
		V0[i] = Boundval[i];
		V0[p+i] = Boundval[p+i];
	}
	
	COO_ordered* A = contract_by_rule(Ba00,V0,create_contraction_rule(&Bijka_Xja));
	for (i=0;i<p;i++) if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
		COO_rank2_set_Dirichlet_symmetric(A,p,i,NULL,0);
	}
	
	double* Jx = COO_rank2_times_vector(A,p,p,&(V0[0]));
	double* Jy = COO_rank2_times_vector(A,p,p,&(V0[p]));
	double* J = zero_vector(2*p+m);
	memcpy(J,Jx,p*sizeof(double));
	memcpy(&(J[p]),Jy,p*sizeof(double));
	
	free(Jx);
	free(Jy);
	return J;
}

void check_Jacobian(mesh_data_2D* mesh,COO_ordered* Ba00,CSC_matrix* J,double* Boundval,double* V0){
	const double h = 1e-8;
	
	int p = mesh->edge_number;
	int m = mesh->el_number;
	int N = 2*p;
	
	double* dX = generate_vector(N,h);
	
	double* X1 = clone_vector(V0,N);
	double* F0 = assembleUpwindFlux2D_RT(mesh,Ba00,Boundval,X1);	
	
	vector_add(X1,dX,1.,N);
	double* F1 = assembleUpwindFlux2D_RT(mesh,Ba00,Boundval,X1);
	
	double* JX = CSCmatrixTimesVector(J,dX);
	
	vector_add(JX,F0,1.0,N);
	vector_add(JX,F1,-1.0,N);
	
	double s = euklid_norm(JX,N)/(h*euklid_norm(F0,N));
	printf("diff: %e\n",s);
	
	free(dX);
	free(X1);
	free(F0);
	free(F1);
	free(JX);
}

CSC_matrix* assembleUpwindFlux2DJacobian_RT(mesh_data_2D* mesh,COO_ordered* Ba00,double* Boundval,double* F,double* V0){
	
	const double tol = 1e-15;
	
	int i;
	
	int p = mesh->edge_number;
	int m = mesh->el_number;
	int offset_yy[2] = {p,p};
	int offset_y0[2] = {p,0};
	
	for (i=0;i<p;i++) if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
		V0[i] = Boundval[i];
		V0[p+i] = Boundval[p+i];
	}
	
	COO_ordered* Axx = contract_by_rule(Ba00,V0,create_contraction_rule(&Bijka_Xja));
	COO_ordered* Ayy = contract_by_rule(Ba00,V0,create_contraction_rule(&Bijka_Xja));
	COO_index_offset(Ayy,offset_yy);
	COO_ordered* Flux = Axx;
	COO_matrix_add(Flux,Ayy,1.);
	COO_free_matrix(&Ayy);

	
	COO_ordered* Axa =  contract_by_rule(Ba00,&(V0[0]),create_contraction_rule(&Bijka_Xk_to_AiJ));
	COO_ordered* Aya =  contract_by_rule(Ba00,&(V0[p]),create_contraction_rule(&Bijka_Xk_to_AiJ));
	COO_index_offset(Aya,offset_y0);
	COO_matrix_add(Flux,Axa,1.);
	COO_matrix_add(Flux,Aya,1.);
	COO_free_matrix(&Axa);
	COO_free_matrix(&Aya);
	
	COO_ordered* BoundMatrix = init_COO(p);
	for (i=0;i<p;i++){		
		if (IsBoundaryEdge(&(mesh->nodes),mesh->edges[i])){
			COO_rank2_set_Dirichlet_symmetric(Flux,2*p,i,BoundMatrix,0);
			COO_rank2_set_Dirichlet_symmetric(Flux,2*p,p+i,BoundMatrix,0);
		}
	}
	double* b = COO_rank2_times_vector(BoundMatrix,2*p,2*p,V0);
	vector_add(F,b,1.,2*p);
	COO_free_matrix(&BoundMatrix);
	free(b);
	
	CSC_matrix* Res = convert_COO_to_CSC(Flux,2*p,2*p,tol);
	//check_Jacobian(mesh,Ba00,Res,Boundval,V0);
	CSC_expand(Res,2*p+m);	
	
	COO_free_matrix(&Axx);
	return Res;
}

double NS_Newton_Step_RT(mesh_data_2D* mesh,CSC_matrix* Linear,COO_ordered* Nonlinear,double* f,double* Boundval,double* X0,double* X){
	
	int p = mesh->edge_number;
	int m = mesh->el_number;
	int N = 2*p+m;
	
	double* b = clone_vector(f,N);
	
	CSC_matrix* Jakobian = assembleUpwindFlux2DJacobian_RT(mesh,Nonlinear,Boundval,b,X0);
	double* Function = assembleUpwindFlux2D_RT(mesh,Nonlinear,Boundval,X0);
	
	CSClinearMap(b,-1.,Linear,X0,N);
	vector_add(b,Function,-1.,N);
	
	CSCmatrixAdd(Jakobian,Linear,N,1.);
	double* dX = zero_vector(N);
	umfpack_solver_col(Jakobian,dX,b);
	vector_add(X,dX,1.,N);
	double res = euklid_norm(dX,2*p);
	shift_Q0_by_mean(mesh,&(X[2*p]));
	
	CSC_free_matrix(&Jakobian);
	free(dX);
	free(b);
	return res;
}

void solveStationaryNS2D(mesh_data_2D* mesh,double* X){
	
	int i;	
	
	const int max_iter = 100;
	const double tol = 1e-10;
	
	double res;
	
	COO_ordered* Nonlinear = NULL;
	int n = mesh->nodes.size;
	int p = mesh->edge_number;
	int m = mesh->el_number;
	int N = 2*p+m;
	
	double* val = zero_vector(2*n);
	for (i=0;i<n;i++){
		val[i] = 0;
		val[n+i] = 0;
	}
	double* Boundval = convertQ1toRT1(mesh,val,2*n);
	free(val);
	
	double* F = NULL;
	CSC_matrix* Linear = NULL;
	
	assembleStationaryStokes2D_RT1Q0(mesh,Boundval,&Linear,&F);
	COO_ordered* Ba00 = assemble_linear_RT_2D(mesh,&linear2D_RT1RT1RT1_a00);
	
	i = 0;
	do{
		Nonlinear = get_upwind_2D(Ba00,X);
		res = NS_Newton_Step_RT(mesh,Linear,Nonlinear,F,Boundval,X,X);
		COO_free_matrix(&Nonlinear);
		i++;
		printf("iterations %d: residuum = %e\n",i,res);
	}while(res>tol && i<max_iter);
	
	free(Boundval);
}



void assemble_Poisson2D(mesh_data_2D* mesh,CSR_matrix** A,double** b){
	const double tol = 1e-8;
	
	int i;
	double x,y;
	
	int n = mesh->nodes.size;
	
	double* F = zero_vector(2*n);
	for (i=0;i<n;i++){
		x = mesh->nodes.Points[i].x;
		y = mesh->nodes.Points[i].y;
		F[i] = sin(2*M_PI*x)*sin(M_PI*y);
		F[i+n] = sin(M_PI*x)*sin(M_PI*y);
	}
	
	COO_ordered* A01 = assemble_linear_2D(mesh,&linear2D_Q1Q1_0a);
	COO_ordered* div = contract_by_rule(A01,NULL,create_contraction_rule(&Aija_to_AiJ));
	*b = COO_rank2_times_vector(div,n,2*n,F);
	
	COO_ordered* I0 = assemble_linear_2D(mesh,&linear2D_Q1_0);	
	COO_ordered* A00 = assemble_linear_2D(mesh,&linear2D_Q1Q1_00);
	COO_ordered* A11 = assemble_linear_2D(mesh,&linear2D_Q1Q1_aa);	
	/* *b = COO_rank2_times_vector(A00,n,n,F);	
	
	for (i=0;i<n;i++){
		if (IsBoundary(&(mesh->nodes),i)>=0){
			COO_rank2_set_Dirichlet(A11,i);
			(*b)[i] = 0;
		}
	}*/
	
	*A = convert_COO_to_CSR(A11,n,tol);
	
	COO_free_matrix(&I0);
	COO_free_matrix(&A00);
	COO_free_matrix(&A11);
}

void solveStokes2DpressureStabilized(mesh_data_2D* mesh,double* X,double dt,int m){
	
	CSR_matrix* As = NULL;
	double* bs = NULL;
	
	CSC_matrix* A = NSassemble_Q1X3_matrix(mesh,&As,dt);
	double* b = NSassemble_Q1X3_inhomogeneity(mesh,&bs,X,dt);
	umfpack_solver_col(A,X,b);
	
	int n = mesh->nodes.size;
	double h = get_longest_edge(&(mesh->nodes));	
	double* R = CSRmatrixTimesVector(As,X);
	vector_add(R,bs,-1.,n);
	
	printf("stab res: %e\n",euklid_norm(R,n)/(h*h*dt));
}

int main(int argc, char* argv[]){
	
	char* Mesh_dir = "/Home/damage/radszuwe/Daten";
	char* Mesh_name = "mini.1";
	
	mesh_data_2D mesh;
	
	read_mesh_2D(Mesh_dir,Mesh_name,&mesh,&Attributes,&attribute_num,CHATTY);		
	int n = mesh.nodes.size;
	int m = mesh.el_number;
	int p = mesh.edge_number;
	int N = 2*p+m;
	
	/*printf("assymbly of system: ...");
	fflush(stdout);
	CSC_matrix* A = NULL;
	double* b = NULL;
	assembleStationaryStokes2D_RT1Q0(&mesh,&A,&b);
	int N = A->cols;
	printf("finished\n");*/
	
	/*printf("solution: ...");
	fflush(stdout);
	double* X = zero_vector(N);
	umfpack_solver_col(A,X,b);
	shift_Q0_by_mean(&mesh,&(X[2*p]));
	printf("finished\n");*/
	
	double* X = generate_vector(N,0);
	solveStationaryNS2D(&mesh,X);
	print_converted(&mesh,X);

	/*double* V = convertRT1toQ1(&mesh,&(X[0]),2*p);
	double* P = convertQ0toQ1(&mesh,&(X[2*p]),m);	
	double* Y = zero_vector(3*n);
	memcpy(Y,P,n*sizeof(double));
	memcpy(&(Y[n]),V,2*n*sizeof(double));
	print_vector(Y,3*n);*/
	
	//print_vector(&V[n],n);
	//print_vector(P,m);
	
	//check_Laplace(&mesh);
	
	/*double* X = zero_vector(2*N+n);
	int i;
	for (i=0;i<n;i++) if (IsBoundary(&(mesh.nodes),i)<0){
		X[i] = 0;
		X[N+i] = 0;
	}
	
	solveStokes2DpressureStabilized(&mesh,X,0.01,100);
	print_vector(&(X[N]),n);*/
	
	/*CSR_matrix* A = NULL;
	double* b = NULL;
	assemble_Poisson2D(&mesh,&A,&b);
	
	double* X = zero_vector(mesh_size);
	umfpack_solver(A,X,b,mesh_size);
	printf("Finished! residuum: %e\n",CSRresiduum(A,b,X));
	print_vector(X,mesh_size);
	 
	free(b);
	free(X);
	CSR_free_matrix(&A);*/
	
	/*COO_ordered* A00 = assemble_linear_2D(&mesh,&linear2D_Q1Q1_10);
	COO_ordered* B000 = assemble_linear_2D(&mesh,&linear2D_Q1Q1Q1_100);
	double* X = generate_vector(mesh_size,1.);
	
	contraction_rule rule = create_contraction_rule(&Bijka_Xk);
	COO_ordered* C00 = contract_by_rule(B000,X,rule);
	
	
	glob_mesh = mesh.nodes;
	set_var_number2D(1);
	//sparse_matrix* a = set_matrix_Aij_00_2D(0,0,&insert_AV);
	//print_sparse(a);
		
	COO_matrix_add(C00,A00,-1.0);
	printf("diff: %e\n",COO_matrix_sum_norm(C00));	
	
	//CSRmatrixAdd(C,A,mesh_size,-1.);
	//printf("diff: %e\n",CSRmatrixNorm(C)/CSRmatrixNorm(A));		

	// clean
	COO_free_matrix(&A00);
	COO_free_matrix(&B000);
	COO_free_matrix(&C00);	
	
	free(X);*/
	
	
	FreeMesh(&(mesh.nodes));												
	free(mesh.elements);
	free(mesh.edges);
}
