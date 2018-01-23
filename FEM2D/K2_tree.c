#include "K2_tree.h"

// global constants

const double Kd_geom_tol = 1e-10;
 
// Code /////////////////////////////////////////// 

static void RB_del_info(void* info){
	int* I = (int*)info;
	free(I);
}

static void RB_del_key(void* key){
	vertex2D** v = (vertex2D**)key;
	free(v);
}

static void RB_print_info(void* info){
	int* I = (int*)info;
	printf("%d\n",*I);
}

static void RB_print_key(const void* key){
	vertex2D** v = (vertex2D**)key;
	printf("(%f,%f)\t",(*v)->coord->x,(*v)->coord->y);
}

int k2_tree_default_cmp(const void* a,const void* b){
	static int dir = RB_X;
	
	if (a==NULL){
		dir = *(int*)b;
		return 0;
	}
	else{
		double A,B;
		vertex2D** va = (vertex2D**)a;
		vertex2D** vb = (vertex2D**)b;
		if (dir==RB_X){
			A = (*va)->coord->x;
			B = (*vb)->coord->x;
		}
		else{
			A = (*va)->coord->y;
			B = (*vb)->coord->y;
		}
		if (fabs(A-B)<Kd_geom_tol) return 0; else return (A>B) ? 1 : -1;
	}
}

point2D* data_functional(vertex2D** Data,int size){
	int i;
	
	point2D* Res = (point2D*)malloc(sizeof(point2D));
	Res->x = 0;
	Res->y = 0;
	for (i=0;i<size;i++){
		Res->x += Data[i]->coord->x;
		Res->y += Data[i]->coord->y;
	}
	Res->x /= (double)size;
	Res->y /= (double)size;
	return Res;
}

static double get_intersection_2D(rb_red_blk_tree* Tree,int size,int dir,vertex2D*** Left,vertex2D*** Right){
	vertex2D** v;
	
	double res = NAN;
	int count = 0;
	int max = size/2;
	
	*Left = (vertex2D**)malloc((size-max)*sizeof(vertex2D*));
	*Right = (vertex2D**)malloc(max*sizeof(vertex2D*));
	rb_red_blk_node* node = RBLargest(Tree);
	while(node!=Tree->nil && node!=NULL){
		v = (vertex2D**)node->key;
		//printf("count: %d\r",count);
		//fflush(stdout);
		
		if (count==max) res = (dir==RB_X) ? (*v)->coord->x : (*v)->coord->y;			//coord P[max] -> intersection
		if (count<max) (*Right)[count] = *v; else (*Left)[count-max] = *v;				//element with index  max -> Left
		count++;
		node = TreePredecessor(Tree,node);
	}
	return res;
}

k2_node* k2_create_node(kd_tree* Tree,vertex2D** Data,int size,int depth){
	int i;
	
	int dir = (depth % 2 == 0) ? RB_X : RB_Y;
	k2_node* Res = (k2_node*)malloc(sizeof(k2_node));
	Res->com = data_functional(Data,size);
	
	if (size>1){
		vertex2D** Left = NULL;
		vertex2D** Right = NULL;	
		
		k2_tree_default_cmp(NULL,(void*)&dir);
		rb_red_blk_tree* Temp = RBTreeCreate(Tree->compare,&RB_del_key,&RB_del_info,&RB_print_key,&RB_print_info);
		for (i=0;i<size;i++) {
			void* key = malloc(sizeof(vertex2D*));
			void* info = malloc(sizeof(int));
			//if (depth % 2 == 0) memcpy(key,(void*)&(Data[i]->coord->x),sizeof(double)); else memcpy(key,(void*)&(Data[i]->coord->y),sizeof(double));
			memcpy(key,(void*)&(Data[i]),sizeof(vertex2D*));
			memcpy(info,(void*)&(Data[i]->index),sizeof(int));
			RBTreeInsert(Temp,key,info);
		}	
		
		Res->intersec = get_intersection_2D(Temp,size,dir,&Left,&Right);
		RBTreeDestroy(Temp);
		Res->left = k2_create_node(Tree,Left,size-(size/2),depth+1);
		Res->right = k2_create_node(Tree,Right,size/2,depth+1);
		Res->leaf_index = -1;
		free(Left);
		free(Right);
	}
	else{
		Res->leaf_index = Data[0]->index;
		Res->intersec = (dir==RB_X) ? Res->com->x : Res->com->y;
		Res->left = NULL;
		Res->right = NULL;
	}
	return Res;
}

void k2_dest_nodes(k2_node* node){
	if (node->left!=NULL){
		k2_dest_nodes(node->left);
		free(node->left);
	}
	if (node->right!=NULL){
		k2_dest_nodes(node->right);
		free(node->right);
	}
	free(node->com);
}

kd_tree* k2_tree_construct(vertex2D** Data,int size,int (*cmp)(const void* a,const void* b)){
	
	int depth = 0;
	kd_tree* Main_tree = (kd_tree*)malloc(sizeof(kd_tree));
	
	Main_tree->compare = cmp;
	Main_tree->root = k2_create_node(Main_tree,Data,size,depth);
	Main_tree->total_size = size;
	
	return Main_tree;
}

void k2_tree_destroy(kd_tree** tree){
	k2_dest_nodes((*tree)->root);
	free(*tree);
}

vertex2D** points_to_vertex_array(point2D* Points,int size){
	int i;
	vertex2D** Res = (vertex2D**)malloc(size*sizeof(vertex2D*));
	for (i=0;i<size;i++){
		Res[i] = (vertex2D*)malloc(sizeof(vertex2D));
		Res[i]->coord = clone_point(&(Points[i]));
		Res[i]->index = i;
	} 
	return Res;
}

void dest_vertex_array(vertex2D*** Array,int size){
	int i;
	for (i=0;i<size;i++){
		free((*Array)[i]->coord);
		free((*Array)[i]);
	}
	free(*Array);
}

k2_node* k2_get_node(kd_tree* Tree,k2_node* Node,point2D* P,int depth){
	if (Node->left==NULL && Node->right==NULL) return Node;
	else{
		double coord = (depth % 2 == 0) ? P->x : P->y;
		if (coord>Node->intersec) return k2_get_node(Tree,Node->right,P,depth+1); 
		else return k2_get_node(Tree,Node->left,P,depth+1); 
	}
} 

vertex2D* k2_approx_nearest_vertex(kd_tree* Tree,vertex2D** Data,point2D* P){
	k2_node* node = k2_get_node(Tree,Tree->root,P,0);
	if (node!=NULL && node->leaf_index>=0) return Data[node->leaf_index]; else return NULL;
}

static int cmp_dist_2D(const void* a,const void* b){
	static point2D sp = {.x=0,.y=0};
	if (a==NULL){
		point2D* P = (point2D*)b;
		sp.x = P->x;
		sp.y = P->y;
		return 0;
	}
	else{
		double da,db;
		point2D* A;
		point2D* B;
		vertex2D** va = (vertex2D**)a;
		vertex2D** vb = (vertex2D**)b;
		A = (*va)->coord;
		B = (*vb)->coord;
		da = sqrdist(A,&sp);
		db = sqrdist(B,&sp);
		if (fabs(da-db)<Kd_geom_tol*Kd_geom_tol) return 0;
		else return (da>db) ? -1 : 1;											// Attention: order reversed ! 
	}
}

static int cmp_vert_index(const void* a,const void* b){
	vertex2D** va = (vertex2D**)a;
	vertex2D** vb = (vertex2D**)b;
	if ((*va)->index==(*vb)->index) return 0;
	else return ((*va)->index>(*vb)->index) ? 1 : -1;
}

int RBTreeInsert_if_not_present(rb_red_blk_tree* tree,void* key,void* info){
	if (RBExactQuery(tree,key)==NULL){
		RBTreeInsert(tree,key,info);
		return 1;
	}
	else return 0;
}

static void InorderMove(rb_red_blk_tree* dest,rb_red_blk_tree* source,rb_red_blk_node* node,size_t key_size){
	if (node!=source->nil){
		InorderMove(dest,source,node->left,key_size);
		InorderMove(dest,source,node->right,key_size);
		
		if (RBTreeInsert_if_not_present(dest,node->key,node->info)==0){
			source->DestroyKey(node->key);
			source->DestroyInfo(node->info);
		}
		free(node);
	}
}
	 
static void RBTree_move_and_destroy(rb_red_blk_tree* dest,rb_red_blk_tree* source) { 
	InorderMove(dest,source,source->root->left,sizeof(vertex2D**));
	source->root->parent = source->nil;
	source->root->left = source->nil;
	source->root->right = source->nil;
	source->root->key = NULL;
	source->root->red = 0;
}

static void RBTree_to_not_ordered_list_helper(rb_red_blk_tree* tree,rb_red_blk_node* node,void*** list,int* size){
	if (node!=tree->nil){
		RBTree_to_not_ordered_list_helper(tree,node->left,list,size);
		(*size)++;
		*list = (void**)realloc(*list,(*size)*sizeof(void*));
		(*list)[(*size)-1] = node->key;
		RBTree_to_not_ordered_list_helper(tree,node->right,list,size);
	}
}

static void RBTree_to_not_ordered_list(rb_red_blk_tree* tree,void*** list,int* size){
	*list = NULL;
	*size = 0;
	RBTree_to_not_ordered_list_helper(tree,tree->root->left,list,size);
}

int find_triangle_with_approx(kd_tree* Tree,vertex2D** Data,mesh2D* Mesh,element_collection* Triangles,int** Tri_index_list,
 int approx_ind,point2D* P){
	int i,j,k,l;
	int* Ind;
	vertex2D* u;
	vertex2D** v;
	vertex2D** w;
	rb_red_blk_node* node;
	
	//int n = Mesh->size;
	//int N = Triangles->size;
	
	int list_size = 0;
	void** List = NULL;
	rb_red_blk_tree* Pool = RBTreeCreate(&cmp_vert_index,&RB_del_key,&RB_del_info,&RB_print_key,&RB_print_info);
	rb_red_blk_tree* Old = RBTreeCreate(&cmp_vert_index,&RB_del_key,&RB_del_info,&RB_print_key,&RB_print_info);
	w = (vertex2D**)malloc(sizeof(vertex2D*));
	*w = Data[approx_ind];
	Ind = (int*)malloc(sizeof(int));
	*Ind = approx_ind;
	RBTreeInsert(Pool,w,Ind);
	
	int index = -1;
	int found = -1;
	while(found<0){
		node = RBLargest(Pool);
		while(node!=Pool->nil && node!=NULL){
			v = (vertex2D**)node->key;
			found = Inside_triangle_set(Mesh,Triangles,P,Tri_index_list[(*v)->index],Kd_geom_tol);
			if (found>=0){
				index = Tri_index_list[(*v)->index][found];				
				break;
			}
			node = TreePredecessor(Pool,node);
		};
		if (found<0){			
			RBTree_to_not_ordered_list(Pool,&List,&list_size);
			RBTree_move_and_destroy(Old,Pool);		
			for (l=0;l<list_size;l++){				
				i = (*(vertex2D**)List[l])->index;
				for (j=0;j<Mesh->Sizes[i];j++){
					k = Mesh->Connections[i][j];
					if (RBExactQuery(Old,(void*)&(Data[k]))==NULL){
						w = (vertex2D**)malloc(sizeof(vertex2D*));
						*w = Data[k];
						Ind = (int*)malloc(sizeof(int));
						*Ind = k;
						if (RBTreeInsert_if_not_present(Pool,w,Ind)==0){
							free(w);
							free(Ind);
						}						
					}
				}
			}
			free(List);			
		}
	}
	
	RBTreeDestroy(Old);
	RBTreeDestroy(Pool);
	return index;
}

static double barycentric_norm(double a,double b,double c){
	double n1 = a*a+b*b+(1.-c)*(1.-c);
	double n2 = a*a+(1.-b)*(1.-b)+c*c;
	double n3 = (1.-a)*(1.-a)+b*b+c*c;
	if (n1<n2) return (n1<n3) ? sqrt(n1) : sqrt(n3);
	else return (n2<n3) ? sqrt(n2) : sqrt(n3);
}

double* get_minimum_interpolation(double* X,mesh2D* Dest,element_collection* Dest_elements,
 mesh2D* Source,element_collection* Source_elements){
																				
	int i,j,k,l;
	double val;
	index3D source_ind,dest_ind;
	point2D bar;
	point2D* A;
	point2D* B;
	point2D* C;
	vertex2D* vertex;
	
	double tol = Kd_geom_tol;
	int M = Dest_elements->size;
	int n = Source->size;
	double* Res = generate_vector(M,DBL_MAX);
	
	// case if meshes are identical
	if (Dest==Source){
		for (k=0;k<M;k++){
			dest_ind = Dest_elements->Elements[k];
			if (X[dest_ind.i]<X[dest_ind.j]) Res[k] = (X[dest_ind.i]<X[dest_ind.k]) ? X[dest_ind.i] : X[dest_ind.k];
			else Res[k] = (X[dest_ind.j]<X[dest_ind.k]) ? X[dest_ind.j] : X[dest_ind.k];
		}
		return Res;
	}
	
	// otherwise
	int** Dest_tri_list = get_triangle_list(Dest,Dest_elements);
	vertex2D** Dest_vertex = points_to_vertex_array(Dest->Points,Dest->size);
	kd_tree* Dest_tree = k2_tree_construct(Dest_vertex,Dest->size,&k2_tree_default_cmp);
	for (i=0;i<n;i++){
		vertex = k2_approx_nearest_vertex(Dest_tree,Dest_vertex,&(Source->Points[i]));		
		l = find_triangle_with_approx(Dest_tree,Dest_vertex,Dest,Dest_elements,Dest_tri_list,vertex->index,&(Source->Points[i]));
		if (l>=0){
			if (fabs(X[i])<Res[l]) Res[l] = fabs(X[i]);
		}
	}	
	k2_tree_destroy(&Dest_tree);
	dest_vertex_array(&Dest_vertex,Dest->size);
	for (j=0;j<Dest->size;j++) free(Dest_tri_list[j]);
	free(Dest_tri_list);
	
	// deal with dest elements having no source points inside
	int** Source_tri_list = get_triangle_list(Source,Source_elements);
	vertex2D** Source_vertex = points_to_vertex_array(Source->Points,Source->size);
	kd_tree* Source_tree = k2_tree_construct(Source_vertex,Source->size,&k2_tree_default_cmp);
	for (i=0;i<M;i++) if (Res[i]==DBL_MAX){
		double a[3],b[3],c[3],val[3];
		dest_ind = Dest_elements->Elements[i];
		for (j=0;j<3;j++){
			switch(j){
				case 0: k = dest_ind.i;break;
				case 1: k = dest_ind.j;break;
				case 2: k = dest_ind.k;break;
			}
			vertex = k2_approx_nearest_vertex(Source_tree,Source_vertex,&(Dest->Points[k]));
			l = find_triangle_with_approx(Source_tree,Source_vertex,Source,Source_elements,Source_tri_list,vertex->index,&(Dest->Points[k]));
				
			if (l<0) source_ind = Source_elements->Elements[get_nearest_triangle(Source,Source_elements,&(Dest->Points[k]))];
			else source_ind = Source_elements->Elements[l];
			
			A = &(Source->Points[source_ind.i]);
			B = &(Source->Points[source_ind.j]);
			C = &(Source->Points[source_ind.k]);
			bar = get_barycentric_2D(A,B,C,&(Dest->Points[k]));
			a[j] = bar.x;
			b[j] = bar.y;
			c[j] = 1.-a[j]-b[j];			
			val[j] = fabs(a[j]*X[source_ind.i]+b[j]*X[source_ind.j]+c[j]*X[source_ind.k]);
			if (l>=0){
				if (a[j]>-tol && b[j]>-tol && c[j]>-tol) Res[i] = (val[j]<Res[i]) ? val[j] : Res[i];
			}
			else Res[i] = (val[j]<Res[i]) ? val[j] : Res[i];
		}
		
		// deal with triangles outside tolerance
		if (Res[i]==DBL_MAX){
			double n1 = barycentric_norm(a[0],b[0],c[0]);
			double n2 = barycentric_norm(a[1],b[1],c[1]);
			double n3 = barycentric_norm(a[2],b[2],c[2]);
			if (n1<n2) k = (n1<n3) ? 0 : 2; else k = (n2<n3) ? 1 : 2;
			Res[i] = val[k];
			//printf("Warning: get_minimum_interpolation\n barycentric coordinates out of tolerance (%e,%e,%e)\n",a[k],b[k],c[k]);
		}
	}
	
	k2_tree_destroy(&Source_tree);
	dest_vertex_array(&Source_vertex,Source->size);
	for (j=0;j<Source->size;j++) free(Source_tri_list[j]);
	free(Source_tri_list);
	return Res;
}

sparse_matrix* get_barycentric_interpolation_matrix(mesh2D* Dest,mesh2D* Source,element_collection* Source_elements){
	int i,tri_ind;
	double a,b,c;
	index3D ind;
	point2D bar;
	point2D* A;
	point2D* B;
	point2D* C;
	vertex2D* vertex;
	
	int m = Dest->size;
	sparse_matrix* Res = sparse_zero(m);
	int** Source_tri_list = get_triangle_list(Source,Source_elements);
	vertex2D** Source_vertex = points_to_vertex_array(Source->Points,Source->size);
	kd_tree* Source_tree = k2_tree_construct(Source_vertex,Source->size,&k2_tree_default_cmp);
	
	for (i=0;i<m;i++){		
		vertex = k2_approx_nearest_vertex(Source_tree,Source_vertex,&(Dest->Points[i]));
		tri_ind = find_triangle_with_approx(Source_tree,Source_vertex,Source,Source_elements,Source_tri_list,vertex->index,&(Dest->Points[i]));
		if (tri_ind<0) tri_ind = get_nearest_triangle(Source,Source_elements,&(Dest->Points[i]));
		
		ind = Source_elements->Elements[tri_ind];			
		A = &(Source->Points[ind.i]);
		B = &(Source->Points[ind.j]);
		C = &(Source->Points[ind.k]);

		bar = get_barycentric_2D(A,B,C,&(Dest->Points[i]));
		a = bar.x;
		b = bar.y;
		c = 1.-a-b;
			
		insert_sparse(Res,a,i,ind.i);
		insert_sparse(Res,b,i,ind.j);
		insert_sparse(Res,c,i,ind.k);
	}
	
	k2_tree_destroy(&Source_tree);
	dest_vertex_array(&Source_vertex,Source->size);
	for (i=0;i<Source->size;i++) free(Source_tri_list[i]);
	free(Source_tri_list);
	remove_zeros(Res);
	return Res;
}

static int insert_helper_list3(index3D ind,int o,int* Res){
	int found = 0;
	if (o!=ind.i && o!=ind.j && o!=ind.k){
		int l = 0;
		while(l<3 && Res[l]>=0){
			l++;
			if (Res[l]==o){
				found = 1;
				break;
				}
			};
		if (!found && l<3) Res[l] = o;				
	}
	return found;
}

int** Triangle_neighbour_map(mesh2D* Mesh,element_collection* Triangles){
	int i,j,k,l,found;
	index3D ind;
	int* list;
	
	int N = Triangles->size;
	int** Res = (int**)malloc(N*sizeof(int*));
	
	for (i=0;i<N;i++){
		Res[i] = (int*)malloc(3*sizeof(int));
		for (j=0;j<3;j++) Res[i][j] = -1;
		ind = Triangles->Elements[i];
		
		for (j=0;j<Mesh->Sizes[ind.i];j++){
			k = Mesh->Connections[ind.i][j];
			list = Overlap(Mesh,ind.i,k);
			insert_helper_list3(ind,list[0],Res[i]);
			if (list[1]>=0) insert_helper_list3(ind,list[1],Res[i]);
			free(list);		
		}		
		for (j=0;j<Mesh->Sizes[ind.j];j++){
			k = Mesh->Connections[ind.j][j];
			list = Overlap(Mesh,ind.j,k);
			insert_helper_list3(ind,list[0],Res[i]);
			if (list[1]>=0) insert_helper_list3(ind,list[1],Res[i]);
			free(list);	
		}		
	}
	return Res;
}

