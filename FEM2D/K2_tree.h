#ifndef K2_tree_H
#define K2_tree_H

#define RB_X 0 
#define RB_Y 1

#include <stdio.h>
#include <math.h>
#include "../FEM2D/FEM2D.h"

// Datentypen /////////////////////////////////////

typedef struct VERTEX2D{
	point2D* coord;
	int index;
}vertex2D;

typedef struct K2_NODE{
	struct K2_NODE* left;
	struct K2_NODE* right;
	point2D* com;
	double intersec;
	int leaf_index;
}k2_node;

typedef struct KD_TREE{
 	k2_node* root;
 	int (*compare)(const void* a,const void* b);
 	int total_size;
}kd_tree;
 
// Funktionen /////////////////////////////////////

kd_tree* k2_tree_construct(vertex2D** Data,int size,int (*cmp)(const void* a,const void* b));
void k2_tree_destroy(kd_tree** tree);

int k2_tree_default_cmp(const void* a,const void* b);
vertex2D** points_to_vertex_array(point2D* Points,int size);
void dest_vertex_array(vertex2D*** Array,int size);
vertex2D* k2_approx_nearest_vertex(kd_tree* Tree,vertex2D** Data,point2D* P);
int find_triangle_with_approx(kd_tree* Tree,vertex2D** Data,mesh2D* Mesh,element_collection* Triangles,int** Tri_index_list,int approx_ind,point2D* P);

double* get_minimum_interpolation(double* X,mesh2D* Dest,element_collection* Dest_elements,
 mesh2D* Source,element_collection* Source_elements);
sparse_matrix* get_barycentric_interpolation_matrix(mesh2D* Dest,mesh2D* Source,element_collection* Source_elements);

#endif
