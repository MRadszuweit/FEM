
#include "FEM2D.h"

// global Variablen //////////////////////////////////////

mesh2D glob_mesh;
element_collection elements;
point2D** Dual_mesh;
double* Dual_areas;
int* Partition = NULL;
static double* Partition_attributes = NULL;		// factors
static int attribute_num = 0;
static int partition_mode = PARTITION_OFF;
static int partition_enabled = 1;
//mesh2D** multi_mesh;
interpolation_map2D interpol2D;

int var_number_2D;
int mesh_number_2D;

// Code ///////////////////////////////////////////////////

int is_valid(point2D p){
	if (p.x==INVALID || p.y==INVALID) return -1;
	else return 1;
}

void set_invalid(point2D* P){
	P->x = INVALID;
	P->y = INVALID;
}

void set_var_number2D(int n){
	var_number_2D = n;
}

int* get_partition(){
	return Partition;
}

int get_var_number2D(){
	return var_number_2D;
}

void enable_partition(){
	partition_enabled = 1;
}

void disable_partition(){
	partition_enabled = 0;
}

void set_partition_mode(int mode){
	if (partition_enabled) partition_mode = mode;
}

void init_partition(int* List,double* Attr,int size){
	Partition = List;
	Partition_attributes = Attr;
	attribute_num = size;
}

void set_part_attr(double* Attr){
	//if (Partition_attributes!=NULL) free(Partition_attributes);
	Partition_attributes = Attr;
}

double get_attr(int i){
	if (!partition_enabled) return 1.;
	if (Partition[i]==0){
		int j,k;
		int count = 0;
		double sum = 0;
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (Partition[k]!=0){
				sum += Partition_attributes[Partition[k]-1];
				count++;
			}
		}
		return (double)sum/count;
	}
	else return Partition_attributes[Partition[i]-1];
}

sparse_matrix* set_attr_matrix(int equ,int var,int varnum){
	int i,index_i,index_j;
	int d = glob_mesh.size;
	double fac;
	sparse_matrix* Res = sparse_zero(varnum*d);
	for (i=0;i<d;i++){
		fac = get_attr(i);
		index_i = d*equ+i;
		index_j = d*var+i;
		insert_sparse(Res,fac,index_i,index_j);
	}
	return Res;
}

void set_mesh_number(int n){
	int i;
	mesh_number_2D = n;
	interpol2D.Triangle_numbers = (int*)malloc(n*sizeof(int));
	interpol2D.Vertex_numbers = (int*)malloc(n*sizeof(int));
	interpol2D.Elements = (index3D***)malloc(n*sizeof(index3D**));
	interpol2D.Map_down = (int**)malloc(n*sizeof(int*));
	interpol2D.Map_up = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++){
		interpol2D.Elements[i] = (index3D**)malloc(0);
		interpol2D.Map_down[i] = NULL;
		interpol2D.Map_up[i] = NULL;
	}
}

void free_interpolation_map2D(){
	int i,j;
	for (i=0;i<mesh_number_2D;i++){
		for (j=0;j<interpol2D.Triangle_numbers[i];j++){
			free(interpol2D.Elements[i][j]);
		}
		free(interpol2D.Elements[i]);
		if (interpol2D.Map_up[i]!=NULL){free(interpol2D.Map_up[i]);}
		if (interpol2D.Map_down[i]!=NULL){free(interpol2D.Map_down[i]);}
	}
	free(interpol2D.Elements);
	free(interpol2D.Map_down);
	free(interpol2D.Map_up);
	free(interpol2D.Triangle_numbers);
	free(interpol2D.Vertex_numbers);
}

double sign(double x){
	if (x<0){return -1;}
	else{return 1;}
}

bool in_array(int i,int j,int n){
	bool res = true;
	if (i<0 || i>=n){res = false;}
	if (j<0 || j>=n){res = false;}
	return res;
}

int index_conversion(int i,int j,int n){
	return i*n+j;
}

index2D inv_index_conversion(int k,int n){
	index2D res;
	res.i = k / n;
	res.j = k % n;
	return res;
}

/*void set_mesh(int n){
	if (n<mesh_number_2D){
		mesh2D* Mesh = multi_mesh[n];
		glob_mesh.size = Mesh->size;
		glob_mesh.Sizes = Mesh->Sizes;
		glob_mesh.Is_boundary = Mesh->Is_boundary;
		glob_mesh.Points = Mesh->Points;
		glob_mesh.Connections = Mesh->Connections;
		glob_mesh.Overlap_table = Mesh->Overlap_table;
	}
}*/

void Null_mesh(mesh2D* Mesh){
	Mesh->Points = NULL;
	Mesh->Connections = NULL;
	Mesh->Sizes = NULL;
	Mesh->Overlap_table = NULL;
	Mesh->size = 0;
}

void init_mesh(int size){
	int i,j,k,l,n,h,counter,sign;
	double x,y;
	double dx = (double) 1/(size-1);
	double dy = (double) dx*sqrt(3)/2;
	glob_mesh.size = size*size;
	glob_mesh.Points = (point2D*) malloc(glob_mesh.size*sizeof(point2D));
	glob_mesh.Connections = (int**) malloc(glob_mesh.size*sizeof(int*));
	glob_mesh.Sizes = (int*) malloc(glob_mesh.size*sizeof(int));
	glob_mesh.Is_boundary = (int*)malloc(glob_mesh.size*sizeof(int));
	//point2D* iterator = glob_mesh.Points;
	for (i=0;i<size;i++){
		for (j=0;j<size;j++){
			int buffer[6];
			k = i;
			l = j;
			sign = -1;
			counter = 0;
			x = (double)i*dx; 
			y = (double)j*dy;
			if (j % 2 != 0){
				x += (double)dx/2;
				sign = 1;
			}
			n = index_conversion(k,l,size);
			glob_mesh.Points[n] = init_point2D(x,y);
			k = i-1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			k = i+1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			k = i;
			l = j-1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			l = j+1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			k = i+sign;
			l = j-1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			l = j+1;
			if (in_array(k,l,size)){
				buffer[counter] = index_conversion(k,l,size);
				counter++;
			}
			glob_mesh.Sizes[n] = counter;
			glob_mesh.Is_boundary[n] = -1;
			int* list = (int*)malloc(counter*sizeof(int));
			for (h=0;h<counter;h++){list[h] = buffer[h];}
			if (counter==0){
				printf("Fehler: Punkt %d hat keine Verbindungen \n",n);
				exit(1);
				}
			else
			{glob_mesh.Connections[n]=list;}
			/*iterator = *init_point2D((double)i/n,(double)-i/n);
			printf("Adresse P: %p\n",iterator);
			//iterator++;*/
		}
	}
}

void pyramid(){
	glob_mesh.size = 5;
	glob_mesh.Points = (point2D*) malloc(glob_mesh.size*sizeof(point2D));
	glob_mesh.Connections = (int**) malloc(glob_mesh.size*sizeof(int*));
	glob_mesh.Sizes = (int*) malloc(glob_mesh.size*sizeof(int));
	glob_mesh.Is_boundary = (int*)malloc(glob_mesh.size*sizeof(int));
	
	glob_mesh.Sizes[0] = 3;
	glob_mesh.Points[0] = init_point2D(0,0);
	glob_mesh.Is_boundary[0] = 1;
	glob_mesh.Connections[0] = (int*)malloc(3*sizeof(int));
	glob_mesh.Connections[0][0] = 1;
	glob_mesh.Connections[0][1] = 4;
	glob_mesh.Connections[0][2] = 2;
	
	glob_mesh.Sizes[1] = 3;
	glob_mesh.Points[1] = init_point2D(1,0);
	glob_mesh.Is_boundary[1] = 0;
	glob_mesh.Connections[1] = (int*)malloc(3*sizeof(int));
	glob_mesh.Connections[1][0] = 0;
	glob_mesh.Connections[1][1] = 4;
	glob_mesh.Connections[1][2] = 3;
	
	glob_mesh.Sizes[2] = 3;
	glob_mesh.Points[2] = init_point2D(0,1);
	glob_mesh.Is_boundary[2] = 0;
	glob_mesh.Connections[2] = (int*)malloc(3*sizeof(int));
	glob_mesh.Connections[2][0] = 0;
	glob_mesh.Connections[2][1] = 4;
	glob_mesh.Connections[2][2] = 3;
	
	glob_mesh.Sizes[3] = 3;
	glob_mesh.Points[3] = init_point2D(1,1);
	glob_mesh.Is_boundary[3] = 2;
	glob_mesh.Connections[3] = (int*)malloc(3*sizeof(int));
	glob_mesh.Connections[3][0] = 1;
	glob_mesh.Connections[3][1] = 4;
	glob_mesh.Connections[3][2] = 2;
	
	glob_mesh.Sizes[4] = 4;
	glob_mesh.Points[4] = init_point2D(0.5,0.5);
	glob_mesh.Is_boundary[4] = 0;
	glob_mesh.Connections[4] = (int*)malloc(4*sizeof(int));
	glob_mesh.Connections[4][0] = 0;
	glob_mesh.Connections[4][1] = 1;
	glob_mesh.Connections[4][2] = 3;
	glob_mesh.Connections[4][3] = 2;
}

void test_mesh(){
	int i,j;
	point2D p;
	for (i=0;i<glob_mesh.size;i++){
		p = glob_mesh.Points[i];
		printf("Koordinaten: %f und %f \n",p.x,p.y);
		printf("Anzahl: %d \n",glob_mesh.Sizes[i]);
		for (j=0;j<glob_mesh.Sizes[i];j++){
			printf("Nachbar: %d\n",glob_mesh.Connections[i][j]);
		}
		if (glob_mesh.Is_boundary[i]>=0){printf("Randpunkt\n");}
	}
	
	/*printf("A00(5,10): %f\n", matrix_Aij_00(5,10));
	point2D v = matrix_Aij_01(5,10);
	printf("A01(5,10): x: %f\n",v.x);
	printf("A01(5,10): y: %f\n",v.y);
	matrix2D A = matrix_Aij_11(5,10);
	printf("A11(5,10): xx: %f und xy: %f \n",A.xx,A.xy);
	printf("A11(5,10): yx: %f und yy: %f \n",A.yx,A.yy);*/
}

void free_mesh(){
	int i,j;
	int n = glob_mesh.size;
	for (i=0;i<n;i++){
		free(glob_mesh.Connections[i]);
	}
	if (glob_mesh.Overlap_table!=NULL){
		for (i=0;i<n;i++){
			for (j=0;j<glob_mesh.Sizes[i];j++){
				free(glob_mesh.Overlap_table[i][j]);
			}
			free(glob_mesh.Overlap_table[i]);
		}
		free(glob_mesh.Overlap_table);
		glob_mesh.Overlap_table = NULL;
	}
	
	if (Dual_mesh!=NULL){
		for (i=0;i<n;i++) free(Dual_mesh[i]);
		free(Dual_mesh);
	}
	if (Dual_areas!=NULL) free(Dual_areas);
	free(glob_mesh.Points);
	free(glob_mesh.Connections);
	free(glob_mesh.Is_boundary);
	free(glob_mesh.Sizes);
	glob_mesh.size = 0;
}

void FreeMesh(mesh2D* Mesh){
	int i,j;
	int n = Mesh->size;
	for (i=0;i<n;i++){
		free(Mesh->Connections[i]);
	}
	if (Mesh->Overlap_table!=NULL){
		for (i=0;i<n;i++){
			for (j=0;j<Mesh->Sizes[i];j++){
				free(Mesh->Overlap_table[i][j]);
			}
			free(Mesh->Overlap_table[i]);
		}
		free(Mesh->Overlap_table);
		Mesh->Overlap_table = NULL;
	}

	free(Mesh->Points);
	free(Mesh->Connections);
	free(Mesh->Is_boundary);
	free(Mesh->Sizes);
	Mesh->size = 0;
}

void free_elements(){
	free(elements.Elements);
}

/*void free_multimeshes(){
	int i,j;
	mesh2D* Mesh;
	for (i=0;i<mesh_number_2D;i++){
		Mesh = multi_mesh[i];
		for (j=0;j<Mesh->size;j++){
			free(Mesh->Connections[j]);
		}
		free(Mesh->Points);
		free(Mesh->Connections);
		free(Mesh->Is_boundary);
		free(Mesh->Sizes);
		free(Mesh);
	}
}*/

void get_mesh_bounds2D(point2D* Min,point2D* Max,mesh2D* Mesh,double tol){
	if (glob_mesh.Points!=NULL){
		int i;
		double x,y,dx,dy;
		double xmin = Mesh->Points[0].x;
		double xmax = Mesh->Points[0].x;
		double ymin = Mesh->Points[0].y;
		double ymax = Mesh->Points[0].y;
		for (i=1;i<Mesh->size;i++){
			x = Mesh->Points[i].x;
			y = Mesh->Points[i].y;
			if (x<xmin){xmin = x;}
			if (x>xmax){xmax = x;}
			if (y<ymin){ymin = y;}
			if (y>ymax){ymax = y;}
		}
		dx = xmax-xmin;
		dy = ymax-ymin;
		Min->x = xmin-dx*tol;
		Min->y = ymin-dy*tol;
		Max->x = xmax+dx*tol;
		Max->y = ymax+dy*tol;
	}
}

int indeq(int ix,int iy,int jx,int jy){
	if (ix==jx && iy==jy){return 1;}
	else {return -1;}
}

/*void seek_whole(int mesh_index,int ind,int mode){
	int i,n;
	point2D a,b,c;
	index3D* Ind;
	point2D p = multi_mesh[mesh_index]->Points[ind];
	n = interpol2D.Triangle_numbers[mesh_index-mode];
	int found = -1;
	for (i=0;i<n;i++){
		Ind = interpol2D.Elements[mesh_index-mode][i];
		a = multi_mesh[mesh_index-mode]->Points[Ind->i];
		b = multi_mesh[mesh_index-mode]->Points[Ind->j];
		c = multi_mesh[mesh_index-mode]->Points[Ind->k];
		if (in_triangle(&a,&b,&c,&p)>0){
			if (mode==UP){interpol2D.Map_up[mesh_index][ind] = i;}
			if (mode==DOWN){interpol2D.Map_down[mesh_index][ind] = i;}
			found = 1;
			break;
		}
	}
	if (found<0){find_nearest_triangle2D(mesh_index,ind,mode);}
}*/

/*void find_nearest_triangle2D(int mesh_index,int ind,int mode){
	int i;
	double r;
	point2D a,b,c,h;
	index3D* Ind;
	point2D p = multi_mesh[mesh_index]->Points[ind];
	int n = interpol2D.Triangle_numbers[mesh_index-mode];
	int min = -1;
	double min_dist = 1E20;
	//printf("nearest_triangle bei index %d in mesh %d\n",ind,mesh_index);
	for (i=0;i<n;i++){
		Ind = interpol2D.Elements[mesh_index-mode][i];
		a = multi_mesh[mesh_index-mode]->Points[Ind->i];
		b = multi_mesh[mesh_index-mode]->Points[Ind->j];
		c = multi_mesh[mesh_index-mode]->Points[Ind->k];
		h = triangle_center(&a,&b,&c);
		r = dist(&h,&p);
		if (r<min_dist){
			min = i;
			min_dist = r;
		}
	}
	if (min<0){
		printf("Fehler in Gitter %d bei Index %d: Interpolation nicht moeglich\n",mesh_index,ind);
		if (mode==UP){interpol2D.Map_up[mesh_index][ind] = -1;}
		if (mode==DOWN){interpol2D.Map_down[mesh_index][ind] = -1;}
		
	}
	else{
		if (mode==UP){interpol2D.Map_up[mesh_index][ind] = min;}
		if (mode==DOWN){interpol2D.Map_down[mesh_index][ind] = min;}
	}
}*/

/*void test_interpol(int mesh_index,int mesh_division){
	int i,n;
	n = multi_mesh[mesh_index+1]->size;
	interpol2D.Map_up[mesh_index+1] = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++){
		seek_whole(mesh_index+1,i,UP);
	}
}*/

/*void create_interpolation_map2D(int mesh_index){
	int i,j,l,n,m,oldsize,ix,iy;
	double dx,dy;
	index3D* Ind;
	point2D a,b,c,p;
	int mesh_division = ceil(sqrt((double)multi_mesh[mesh_index]->size)/5);
	printf("interpolation mesh resolution %d: %d\n",mesh_index+1,mesh_division);
	if (mesh_index<mesh_number_2D-1){
		point2D min = init_point2D(0,0);
		point2D max = init_point2D(0,0);
		get_mesh_bounds2D(&min,&max,multi_mesh[mesh_index]);
		dx = (max.x-min.x)/mesh_division;
		dy = (max.y-min.y)/mesh_division;
		n = interpol2D.Triangle_numbers[mesh_index];
		int*** Element_list = (int***)malloc(mesh_division*sizeof(int**));
		int** Sizes_list = (int**)malloc(mesh_division*sizeof(int*));
		for (i=0;i<mesh_division;i++){
			Element_list[i] = (int**)malloc(mesh_division*sizeof(int*));
			Sizes_list[i] = (int*)malloc(mesh_division*sizeof(int));
			for (j=0;j<mesh_division;j++){
				Element_list[i][j] = (int*)malloc(0);
				Sizes_list[i][j] = 0;
			}
		}
		int kx[3];
		int ky[3];
		int take[3];
		for (i=0;i<n;i++){
			Ind = interpol2D.Elements[mesh_index][i];
			a = multi_mesh[mesh_index]->Points[Ind->i];
			b = multi_mesh[mesh_index]->Points[Ind->j];
			c = multi_mesh[mesh_index]->Points[Ind->k];
			kx[0] = floor((a.x-min.x)/dx);
			ky[0] = floor((a.y-min.y)/dy);
			if (kx[0]>=mesh_division || ky[0]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			take[0] = 1;
			kx[1] = floor((b.x-min.x)/dx);
			ky[1] = floor((b.y-min.y)/dy);
			if (kx[1]>=mesh_division || ky[1]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			if (indeq(kx[1],ky[1],kx[0],ky[0])>0){take[1] = -1;}else{take[1] = 1;}
			kx[2] = floor((c.x-min.x)/dx);
			ky[2] = floor((c.y-min.y)/dy);
			if (kx[2]>=mesh_division || ky[2]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			if (indeq(kx[2],ky[2],kx[0],ky[0])>0 || indeq(kx[2],ky[2],kx[1],ky[1])>0){take[2] = -1;}else{take[2] = 1;}
			for (l=0;l<3;l++){
				if (take[l]>0){
					oldsize = Sizes_list[kx[l]][ky[l]];
					Element_list[kx[l]][ky[l]] = (int*)realloc(Element_list[kx[l]][ky[l]],(oldsize+1)*sizeof(int));
					Element_list[kx[l]][ky[l]][oldsize] = i;
					Sizes_list[kx[l]][ky[l]] = oldsize+1;
				}
			}
		}
		int found;
		m = multi_mesh[mesh_index+1]->size;
		interpol2D.Map[mesh_index+1] = (int*)malloc(m*sizeof(int));
		for (i=0;i<m;i++){
			p = multi_mesh[mesh_index+1]->Points[i];
			ix = floor((p.x-min.x)/dx);
			iy = floor((p.y-min.y)/dy);
			found = -1;
			for (j=0;j<Sizes_list[ix][iy];j++){
				l = Element_list[ix][iy][j];
				Ind = interpol2D.Elements[mesh_index][l];
				a = multi_mesh[mesh_index]->Points[Ind->i];
				b = multi_mesh[mesh_index]->Points[Ind->j];
				c = multi_mesh[mesh_index]->Points[Ind->k];
				if (in_triangle(&a,&b,&c,&p)>0){
					interpol2D.Map[mesh_index+1][i] = l;
					found = 1;
					break;
				}
			}
			if (found<0){seek_whole(mesh_index+1,i);}
		}
		for (i=0;i<mesh_division;i++){
			for (j=0;j<mesh_division;j++){free(Element_list[i][j]);}
			free(Element_list[i]);
			free(Sizes_list[i]);
		}
		free(Element_list);
		free(Sizes_list);
	}
}*/

/*void create_interpolation_map2D(int mesh_index,int mode){
	int i,j,l,n,m,oldsize,ix,iy;
	double dx,dy;
	index3D* Ind;
	point2D a,b,c,p;
	int mesh_division = ceil(sqrt((double)multi_mesh[mesh_index]->size)/5);	
	if (mesh_index+mode<mesh_number_2D && mesh_index+mode>=0){
		printf("interpolation mesh resolution %d: %d\n",mesh_index+mode,mesh_division);
		point2D min = init_point2D(0,0);
		point2D max = init_point2D(0,0);
		get_mesh_bounds2D(&min,&max,multi_mesh[mesh_index]);
		dx = (max.x-min.x)/mesh_division;
		dy = (max.y-min.y)/mesh_division;
		n = interpol2D.Triangle_numbers[mesh_index];
		int*** Element_list = (int***)malloc(mesh_division*sizeof(int**));
		int** Sizes_list = (int**)malloc(mesh_division*sizeof(int*));
		for (i=0;i<mesh_division;i++){
			Element_list[i] = (int**)malloc(mesh_division*sizeof(int*));
			Sizes_list[i] = (int*)malloc(mesh_division*sizeof(int));
			for (j=0;j<mesh_division;j++){
				Element_list[i][j] = (int*)malloc(0);
				Sizes_list[i][j] = 0;
			}
		}
		int kx[3];
		int ky[3];
		int take[3];
		for (i=0;i<n;i++){
			Ind = interpol2D.Elements[mesh_index][i];
			a = multi_mesh[mesh_index]->Points[Ind->i];
			b = multi_mesh[mesh_index]->Points[Ind->j];
			c = multi_mesh[mesh_index]->Points[Ind->k];
			kx[0] = floor((a.x-min.x)/dx);
			ky[0] = floor((a.y-min.y)/dy);
			if (kx[0]>=mesh_division || ky[0]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			take[0] = 1;
			kx[1] = floor((b.x-min.x)/dx);
			ky[1] = floor((b.y-min.y)/dy);
			if (kx[1]>=mesh_division || ky[1]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			if (indeq(kx[1],ky[1],kx[0],ky[0])>0){take[1] = -1;}else{take[1] = 1;}
			kx[2] = floor((c.x-min.x)/dx);
			ky[2] = floor((c.y-min.y)/dy);
			if (kx[2]>=mesh_division || ky[2]>=mesh_division){printf("Fehler bei Index %d\n",i);}
			if (indeq(kx[2],ky[2],kx[0],ky[0])>0 || indeq(kx[2],ky[2],kx[1],ky[1])>0){take[2] = -1;}else{take[2] = 1;}
			for (l=0;l<3;l++){
				if (take[l]>0){
					oldsize = Sizes_list[kx[l]][ky[l]];
					Element_list[kx[l]][ky[l]] = (int*)realloc(Element_list[kx[l]][ky[l]],(oldsize+1)*sizeof(int));
					Element_list[kx[l]][ky[l]][oldsize] = i;
					Sizes_list[kx[l]][ky[l]] = oldsize+1;
				}
			}
		}
		int found;
		m = multi_mesh[mesh_index+mode]->size;
		if (mode==UP){interpol2D.Map_up[mesh_index+mode] = (int*)malloc(m*sizeof(int));}
		if (mode==DOWN){interpol2D.Map_down[mesh_index+mode] = (int*)malloc(m*sizeof(int));}
		for (i=0;i<m;i++){
			p = multi_mesh[mesh_index+mode]->Points[i];
			ix = floor((p.x-min.x)/dx);
			iy = floor((p.y-min.y)/dy);
			found = -1;
			for (j=0;j<Sizes_list[ix][iy];j++){
				l = Element_list[ix][iy][j];
				Ind = interpol2D.Elements[mesh_index][l];
				a = multi_mesh[mesh_index]->Points[Ind->i];
				b = multi_mesh[mesh_index]->Points[Ind->j];
				c = multi_mesh[mesh_index]->Points[Ind->k];
				if (in_triangle(&a,&b,&c,&p)>0){
					if (mode==UP){interpol2D.Map_up[mesh_index+mode][i] = l;}
					if (mode==DOWN){interpol2D.Map_down[mesh_index+mode][i] = l;}
					found = 1;
					break;
				}
			}
			if (found<0){seek_whole(mesh_index+mode,i,mode);}
		}
		for (i=0;i<mesh_division;i++){
			for (j=0;j<mesh_division;j++){free(Element_list[i][j]);}
			free(Element_list[i]);
			free(Sizes_list[i]);
		}
		free(Element_list);
		free(Sizes_list);
	}
}*/

void segment_tree_del_key(void* Info){
	slab_tree* Local_tree = (slab_tree*)Info;
	RBTreeDestroy(Local_tree->Tree);
	free(Local_tree);
}

void local_tree_del_key(void* Info){
	double* Y = (double*)Info;
	free(Y);
}

void tree_del_info(void* Key){											
	int* Num = (int*)Key;
	free(Num);
}

void dummy(void* a){};

int segment_compare(const void* A,const void* B){
	const double tol = 1e-10;
	
	double* A_top = (double*)A;
	double* B_top = (double*)B;
	if (!isnan(A_top[2])){
		if (fabs(A_top[1]-B_top[1])<tol) return (A_top[2]>B_top[2]) ? 1: -1;
		else return (A_top[1]>B_top[1]) ? 1: -1;
	}
	else{
		double ys = B_top[1]+B_top[2]*(A_top[0]-B_top[0]);
		if (fabs(A_top[1]-ys)<tol) return 0; else return (A_top[1]>ys) ? 1 : -1;
	}
}


int slab_compare(const void* A,const void* B){
	const double tol = 1e-10;
	
	slab_tree* TA = (slab_tree*)A;
	slab_tree* TB = (slab_tree*)B;
	if (fabs(TA->x-TB->x)<tol){
		if (fabs(TA->y-TB->y)<tol) return 0;
		else return ((TA->y)>(TB->y)) ? 1 : -1;
	}
	else return ((TA->x)>(TB->x)) ? 1 : -1;
}

static FILE* globfile = NULL;

void print_tree_to_file(rb_red_blk_tree* Tree){
	globfile = fopen("/Home/damage/radszuwe/Daten/tree","w");
	RBTreePrint(Tree);
	fclose(globfile);
}

void Printkey(const void* Key){
	if (globfile!=NULL){
		slab_tree* Tree = (slab_tree*)Key;
		fprintf(globfile,"slab: %f\t %f\n",Tree->x,Tree->y);
		fprintf(globfile,"segments: %d",RBTreeCount(Tree->Tree));
		//RBTreePrint(Tree->Tree);
		fprintf(globfile,"\n");
	}
}

void SubPrintinfo(void* Info){
	int* I = (int*)Info;
	if (globfile!=NULL) fprintf(globfile,"%d)\t",*I);
	//printf("%d\t",*I);
}

void SubPrintkey(const void* Info){
	double* Y = (double*)Info;
	if (globfile!=NULL) fprintf(globfile,"(x=%f,y=%f,a=%f,",Y[0],Y[1],Y[2]);
	//printf("(x=%f,y=%f,a=%f)\n",Y[0],Y[1],Y[2]);
}

void Printinfo(void* Info){
	int* I = (int*)Info;
	printf("ind: %d\n",*I);
}

int stack_size(stk_stack* Stack){
	
	int counter = 0;
	stk_stack_node* current = Stack->top;
	stk_stack_node* end = Stack->tail;		
	if (current==NULL) return counter;
	
	counter++;
	while(current!=end){
		current = current->next;
		counter++;
	};
	return counter;
}

void print_stack(stk_stack* Stack){
	
	stk_stack_node* current = Stack->top;
	stk_stack_node* end = Stack->tail;		
	if (current==NULL) return;
	
	slab_tree* Slab = (slab_tree*)((rb_red_blk_node*)current->info)->key;
	printf("%f\t%f\n",Slab->x,Slab->y);
	
	while(current!=end){
		current = current->next;
		Slab = (slab_tree*)((rb_red_blk_node*)current->info)->key;
		printf("%f\t%f\n",Slab->x,Slab->y);
	};
	printf("\n");	
}

rb_red_blk_tree* slab_search_tree(mesh2D* Mesh,element_collection* Triangles,int* end_tri_index){
	const double tol = 1e-10;
	
	int i,j,k;
	double y1,y2,left,right,x_left,y_left,x_right,y_right,local_left,local_right;
	int* Index;
	slab_tree* Key;
	index3D tri_ind;
	index2D ind;
	
	int n = Mesh->size;
	int N = Triangles->size;
	
	// init slab tree
	rb_red_blk_tree* Main_tree = RBTreeCreate(&slab_compare,&segment_tree_del_key,&tree_del_info,&Printkey,&Printinfo);
	for (i=0;i<n;i++){
		Index = (int*)malloc(sizeof(int));
		*Index = i;
		Key = (slab_tree*)malloc(sizeof(slab_tree));
		Key->x = Mesh->Points[i].x;
		Key->y = Mesh->Points[i].y;
		Key->Tree = RBTreeCreate(&segment_compare,&local_tree_del_key,&tree_del_info,&SubPrintkey,&SubPrintinfo);		
		RBTreeInsert(Main_tree,Key,Index);
	}
	
	// insert segments	
	int end_node_index = 0;
	slab_tree end_node_point = {.Tree=NULL,.x=Mesh->Points[Triangles->Elements[0].i].x,.y=Mesh->Points[Triangles->Elements[0].i].y};
	for (j=0;j<N;j++){
		tri_ind = Triangles->Elements[j];
		for (k=0;k<3;k++){
			switch(k){
				case 0: 
					ind.i = tri_ind.i;
					ind.j = tri_ind.j;
					break;
				case 1: 
					ind.i = tri_ind.j;
					ind.j = tri_ind.k;
					break;
				case 2:
					ind.i = tri_ind.k;
					ind.j = tri_ind.i;
					break;				
			}
		
			x_left = Mesh->Points[ind.i].x;
			x_right = Mesh->Points[ind.j].x;
			y_left = Mesh->Points[ind.i].y;
			y_right = Mesh->Points[ind.j].y;		
			slab_tree lower = {.Tree=NULL,.x=x_left,.y=y_left};
			slab_tree upper = {.Tree=NULL,.x=x_right,.y=y_right};
			if (slab_compare(&upper,&lower)<0){
				lower.x = x_right;
				lower.y = y_right;
				upper.x = x_left;
				upper.y = y_left;
			}
			
			if (slab_compare(&end_node_point,&upper)<0){
				end_node_point.x = upper.x;
				end_node_point.y = upper.y;
				end_node_index = j;
			}
			
			double a;
			if (fabs(upper.x-lower.x)>tol) a = (upper.y-lower.y)/(upper.x-lower.x); else a = (upper.y>lower.y)/tol;
			
			stk_stack* Slab_list = RBEnumerate(Main_tree,(void*)&lower,(void*)&upper);
			if (stack_size(Slab_list)<2){
				printf("something's wrong at index %d,%d -> abort\n",j,k);
				print_stack(Slab_list);
				exit(0);
			}
			
			stk_stack_node* next;
			stk_stack_node* current = Slab_list->top;
			stk_stack_node* end = Slab_list->tail;			
			do{
				slab_tree* Local_tree = (slab_tree*)((rb_red_blk_node*)current->info)->key;
				local_left = Local_tree->x;
				next = current->next;											  
				if (next!=NULL){
					slab_tree* Right_tree = (slab_tree*)((rb_red_blk_node*)next->info)->key;					
					local_right = Right_tree->x;
					// set triangle number
					int* J = (int*)malloc(sizeof(int));
					*J = j;
					// set segment coordinates and direction 
					double* Segment = (double*)malloc(3*sizeof(double));		// left x,left y, a=dy/dx
					Segment[0] = local_left;
					Segment[1] = (fabs(upper.x-lower.x)>tol) ? lower.y+a*(local_left-lower.x) : lower.y;
					Segment[2] = a;
					
					RBTreeInsert(Local_tree->Tree,Segment,J);
					current = next;
				}
			}while(current!=end && next!=NULL);	
			StackDestroy(Slab_list,&dummy);		
		}	
	}
	*end_tri_index = end_node_index;
	return Main_tree;
}

rb_red_blk_node* get_largest_left(rb_red_blk_tree* tree,rb_red_blk_node* node,void* key){
	
	if (node==tree->nil) return NULL;
	else{		
		tree->PrintKey(node->key);
		int cmp = tree->Compare(key,node->key);
		if (cmp==1){
			rb_red_blk_node* maxleft = get_largest_left(tree,node->right,key);
			if (maxleft==NULL) return node; 
			else return maxleft;			
		}
		else if (cmp==-1){
			return get_largest_left(tree,node->left,key);					
		}
		else return node;
	}
}

rb_red_blk_node* red_blk_get_largest_left_node(rb_red_blk_tree* tree,rb_red_blk_node* node,void* key){
	return get_largest_left(tree,tree->root->left,key);	
}

slab_tree* get_left_slab_tree(rb_red_blk_tree* Main_tree,point2D* P){	
	slab_tree point = {.Tree=NULL,.x=P->x,.y=P->y};
	rb_red_blk_node* node = get_largest_left(Main_tree,Main_tree->root->left,&point);
	if (node!=NULL) return (slab_tree*)node->key; else return NULL;
}

int get_max_segment(slab_tree* Local_tree,point2D* P){
	if (Local_tree->Tree!=NULL){
		double* Key = (double*)malloc(3*sizeof(double));
		Key[0] = P->x;
		Key[1] = P->y;
		Key[2] = NAN;

		rb_red_blk_node* node = red_blk_get_largest_left_node(Local_tree->Tree,Local_tree->Tree->root->left,Key);
		
		
		free(Key);
		return *(int*)node->info;
	}
	return -1;
}

int get_triangle_index(search_tree2D* Search_tree,point2D* P){
	const double tol = 1e-10;
	
	int tri_index;
	
	slab_tree* Local_tree = get_left_slab_tree(Search_tree->Main_tree,P);
	if (Local_tree!=NULL){
		double* Key = (double*)malloc(3*sizeof(double));
		Key[0] = P->x;
		Key[1] = P->y;
		Key[2] = NAN;
		
		rb_red_blk_node* node = red_blk_get_largest_left_node(Local_tree->Tree,Local_tree->Tree->root->left,Key);
		free(Key);
		if (node!=NULL) tri_index = *(int*)node->info; 
		else{
			if (Inside_triangle(Search_tree->Mesh,Search_tree->Triangles,P,Search_tree->last_index,tol)) return Search_tree->last_index; else return -1;	
		}
		if (Inside_triangle(Search_tree->Mesh,Search_tree->Triangles,P,tri_index,tol)) return tri_index;
		else{
			rb_red_blk_node* lower = TreePredecessor(Local_tree->Tree,node);
			if (lower!=Local_tree->Tree->nil){
				tri_index = *(int*)lower->info;
				if (Inside_triangle(Search_tree->Mesh,Search_tree->Triangles,P,tri_index,tol)) return tri_index;
			}
			rb_red_blk_node* upper = TreeSuccessor(Local_tree->Tree,node);
			if (upper!=Local_tree->Tree->nil){
				tri_index = *(int*)upper->info;
				if (Inside_triangle(Search_tree->Mesh,Search_tree->Triangles,P,tri_index,tol)) return tri_index;				
			}
		}
	}
	return -1;
}

search_tree2D* create_search_tree2D(mesh2D* Mesh,element_collection* Triangles){
	search_tree2D* Res = (search_tree2D*)malloc(sizeof(search_tree2D));
	Res->Mesh = Mesh;
	Res->Triangles = Triangles;
	int last_index = -1;
	rb_red_blk_tree* Main_tree = slab_search_tree(Mesh,Triangles,&last_index);
	Res->Main_tree = Main_tree;
	Res->last_index = last_index;
	return Res;
}

void free_search_tree2D(search_tree2D** Tree){										// does not free the mesh ! 
	RBTreeDestroy((*Tree)->Main_tree);
	free(*Tree);
	*Tree = NULL;
}

double linear_interpolation2D(point2D* P1,point2D* P2,point2D* P3
	,double u1,double u2,double u3,point2D* P){
	double detF = det(P1,P2,P3);
	double a = ((u3-u2)*P1->y+(u1-u3)*P2->y+(u2-u1)*P3->y)/detF;
	double b = ((u2-u3)*P1->x+(u3-u1)*P2->x+(u1-u2)*P3->x)/detF;
	double c = ((P2->x*P3->y-P3->x*P2->y)*u1+(P3->x*P1->y-P1->x*P3->y)*u2
				+(P1->x*P2->y-P2->x*P1->y)*u3)/detF;
	return a*P->x+b*P->y+c;
}

point2D triangle_center(point2D* P1,point2D* P2,point2D* P3){
	point2D res = init_point2D(0,0);
	double fac = (double)1/3;
	vec_add_mult(&res,P1,fac);
	vec_add_mult(&res,P2,fac);
	vec_add_mult(&res,P3,fac);
	return res;
}

/*void interpolation2D(double* Dest,double* Source,int mesh_index,int varnum,int mode){
	int i,j,n,d,k;
	double u_a,u_b,u_c;
	index3D* Ind;
	point2D a,b,c,p;
	if (mesh_index-mode>=0 && mesh_index-mode<mesh_number_2D){
		n = multi_mesh[mesh_index]->size;
		d = multi_mesh[mesh_index-mode]->size;
		for (i=0;i<n;i++){
			if (mode==UP){k = interpol2D.Map_up[mesh_index][i];}
			if (mode==DOWN){k = interpol2D.Map_down[mesh_index][i];}
			Ind = interpol2D.Elements[mesh_index-mode][k];
			a = multi_mesh[mesh_index-mode]->Points[Ind->i];
			b = multi_mesh[mesh_index-mode]->Points[Ind->j];
			c = multi_mesh[mesh_index-mode]->Points[Ind->k];
			p = multi_mesh[mesh_index]->Points[i];
			for (j=0;j<varnum;j++){
				u_a = Source[j*d+Ind->i];
				u_b = Source[j*d+Ind->j];
				u_c = Source[j*d+Ind->k];
				Dest[j*n+i] = linear_interpolation2D(&a,&b,&c,u_a,u_b,u_c,&p);			
			}
		}
	}
}*/

int get_center_index(){
	int i,res,n;
	double min,d;
	n = glob_mesh.size;
	point2D center;
	center.x = 0;
	center.y = 0;
	for (i=0;i<n;i++){
		vec_add_mult(&center,&glob_mesh.Points[i],1);
	}
	vec_mult(&center,(double)1/n);
	res = 0;
	min = dist(&center,&glob_mesh.Points[0]);
	for (i=1;i<n;i++){
		d = dist(&center,&glob_mesh.Points[i]);
		if (d<min){
			min = d;
			res = i;
		}
	}
	if (glob_mesh.Is_boundary[res]>=0){
		printf("Fehler: Schwerpunktindex %d liegt am Rand !\n",res);
		return -1;
	}
	else{return res;}
}

point2D grad_2D(int i,int j,int k){						// f(i)=1, f(j)=0, f(k)=0
	point2D res;
	point2D* Pi = &(glob_mesh.Points[i]);
	point2D* Pj = &(glob_mesh.Points[j]);
	point2D* Pk = &(glob_mesh.Points[k]);
	double det = Pk->x*(-Pi->y + Pj->y)+Pj->x*(Pi->y-Pk->y)+Pi->x*(-Pj->y+Pk->y);
	res.x = (-Pj->y+Pk->y)/det;
	res.y = (Pj->x-Pk->x)/det;
	return res;
}

point2D bound_line_2D(int i,int j,int k){				// i,j at boundary, k inner node
	double d = dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]));
	point2D res = get_normal(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]),&(glob_mesh.Points[k]));
	vec_mult(&res,d);
	return res;
}

/*! 
 *  @param i,j: Knotennummer 
 *  @return Gibt eine zweikomponentige Liste mit den Indizes der Ergänzungspunkte
 * zu Dreiecken aus. Liegt ein Randpunkt vor, so ist das zweite Element -1.
*/

int* overlap(int i,int j){
	int k,l,n;
	int counter = 0;
	int* Res = (int*)malloc(2*sizeof(int));
	if (glob_mesh.Overlap_table==NULL){
		Res[0] = -1;
		Res[1] = -1;
		for (k=0;k<glob_mesh.Sizes[i];k++){
			n = glob_mesh.Connections[i][k];
			for (l=0;l<glob_mesh.Sizes[j];l++){
				if  (glob_mesh.Connections[j][l]==n){
					Res[counter] = n;
					counter++;
				}
			}
		}
		if (counter > 2 || counter==0){printf("Verbindungsfehler in Knoten %d, size: %d\n",i,glob_mesh.size);}
		return Res;
	}
	else{
		for (k=0;k<glob_mesh.Sizes[i];k++){
			if (glob_mesh.Overlap_table[i][k][0]==j){
				Res[0] = glob_mesh.Overlap_table[i][k][1];
				Res[1] = glob_mesh.Overlap_table[i][k][2];
				return Res;
			}
		}
		Res[0] = -1;
		Res[1] = -1;
		printf("Fehler in Look-up-table bei Knoten %d\n",i);
		return Res;
	}
}

int* init_overlap(mesh2D* Mesh,int i,int j){
	int k,l,n;
	int counter = 0;
	int* Res = (int*)malloc(2*sizeof(int));
	Res[0] = -1;
	Res[1] = -1;
	for (k=0;k<Mesh->Sizes[i];k++){
		n = Mesh->Connections[i][k];
		for (l=0;l<Mesh->Sizes[j];l++){
			if  (Mesh->Connections[j][l]==n){
				Res[counter] = n;
				counter++;
			}
		}
	}
	if (counter > 2 || counter==0){printf("Verbindungsfehler in Knoten %d\n",i);}
	return Res;
}

void get_boundary_elements(int** Indices,int* len){
	int i,j;
	index3D ind;
	int n = elements.size;
	*Indices = NULL;
	*len = 0;
	
	for (i=0;i<n;i++){
		ind = elements.Elements[i];
		if (is_boundary(ind.i)>=0 || is_boundary(ind.j)>=0 || is_boundary(ind.k)>=0){
			(*len)++;
			*Indices = (int*)realloc(*Indices,(*len)*sizeof(int));
			(*Indices)[(*len)-1] = i;
		}
	}
}

int get_boundary_edge_num(mesh2D* Mesh){
	int i;
	
	int b = 0;
	for (i=0;i<Mesh->size;i++) if (IsBoundary(Mesh,i)>=0) b++;
	
	return b;
}

/*int isBoundaryEdge(mesh2D* Mesh,int i,int j){
	if (IsBoundary(Mesh,i)>=0 && IsBoundary(Mesh,j)>=0) return 1; else return 0;
}*/

double bound_curvature(int i){
	if (is_boundary(i)>=0){
		int j,k;
		double s,res;
		point2D curv;
		int* Bound = NULL;
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (is_boundary(k)>=0){
				if (Bound==NULL){
					Bound = (int*)malloc(2*sizeof(int));
					Bound[0] = k;
				}
				else Bound[1] = k;				
			} 
		}
		if (Bound!=NULL){
			int* list0 = overlap(Bound[0],i);
			int* list1 = overlap(i,Bound[1]);
			point2D* Inner = clone_point(&(glob_mesh.Points[list0[0]]));
			vec_add_mult(Inner,&(glob_mesh.Points[list1[0]]),1.);
			vec_mult(Inner,(double)1/2);
			vec_add_mult(Inner,&glob_mesh.Points[i],-1.);
			
			point2D* P0 = clone_point(&(glob_mesh.Points[i]));
			vec_add_mult(P0,&(glob_mesh.Points[Bound[0]]),-1.);
			point2D* P1 = clone_point(&(glob_mesh.Points[Bound[1]]));
			vec_add_mult(P1,&(glob_mesh.Points[i]),-1.);
			s = (vec_abs(P0)+vec_abs(P1))/2.;
			normalize(P0);
			normalize(P1);
			curv = vec_diff(P1,P0);
			res = vec_abs(&curv)/s;
			if (vec_scalar(&curv,Inner)<0) res *= -1.;
			
			free(list0);
			free(list1);
			free(Inner);
			free(P0);
			free(P1);
			free(Bound);
			return res;
		}
		else{
			printf("bound_curvature: error in mesh at index %d\n",i);
			exit(0);
		}		
	}
	else return NAN;
}

void edge_refine(point2D** Poly,int* Loop_start,int loop_num,double threshold,double dmin,double edge_refine_fraction){			// refines poly data according to bound curvature
																									// Loop_start[loop_num] = total_length
	int i,j,k,order1,order2;
	double angle,d,s_prev,s_curr;
	point2D n_prev,n_curr,p1,p2;
	
	for (i=0;i<loop_num;i++){
		j = Loop_start[i];
		n_prev = init_point2D((*Poly)[j].x,(*Poly)[j].y);
		vec_add_mult(&n_prev,&((*Poly)[Loop_start[i+1]-1]),-1.);
		s_prev = vec_abs(&n_prev);
		normalize(&n_prev);

		while(j<Loop_start[i+1]){
			if (j==Loop_start[i+1]-1) n_curr = init_point2D((*Poly)[Loop_start[i]].x,(*Poly)[Loop_start[i]].y);
			else n_curr = init_point2D((*Poly)[j+1].x,(*Poly)[j+1].y);
			vec_add_mult(&n_curr,&((*Poly)[j]),-1.);
			s_curr = vec_abs(&n_curr);
			normalize(&n_curr);
			
			
			angle = acos(vec_scalar(&n_curr,&n_prev));
			if (fabs(angle)>threshold){
				order1 = (int)round(-log(dmin/s_prev)/log(2.));
				order2 = (int)round(-log(dmin/s_curr)/log(2.));
				if (order1<0) order1 = 0;
				if (order2<0) order2 = 0;
				
				// shift data
				for (k=i+1;k<=loop_num;k++) Loop_start[k] += order1+order2;				
				*Poly = (point2D*)realloc(*Poly,Loop_start[loop_num]*sizeof(point2D));
				for (k=Loop_start[loop_num]-1;k>j+order1+order2;k--){
					(*Poly)[k] = init_point2D((*Poly)[k-order1-order2].x,(*Poly)[k-order1-order2].y);
				}
				(*Poly)[j+order1] = init_point2D((*Poly)[j].x,(*Poly)[j].y);
				
				// create new points
				if (order1>0){
					d = 0.5;
					p1 = init_point2D((*Poly)[j+order1].x,(*Poly)[j+order1].y);
					vec_add_mult(&p1,&n_prev,-d*s_prev);	
					(*Poly)[j+0] = init_point2D(p1.x,p1.y);
					for (k=1;k<order1;k++){
						d *= edge_refine_fraction;					
						vec_add_mult(&p1,&n_prev,d*s_prev);	
						(*Poly)[j+k] = init_point2D(p1.x,p1.y);
						//printf("new -> %d\n",j+k);
					}
				}
				if (order2>0){				
					d = s_prev/s_curr*exp((double)log(2.)*(order2-order1-1));
					if (d>1.) d = 0.75;
					p2 = init_point2D((*Poly)[j+order1].x,(*Poly)[j+order1].y);
					vec_add_mult(&p2,&n_curr,d*s_curr);
					(*Poly)[j+order1+order2] = init_point2D(p2.x,p2.y);				
					for (k=1;k<order2;k++){
						d *= edge_refine_fraction;										
						vec_add_mult(&p2,&n_curr,-d*s_curr);	
						(*Poly)[j+order1+order2-k] = init_point2D(p2.x,p2.y);
						//printf("new -> %d\n",j+order1+order2-k);
					}
					
				}
				
				j += order1+order2+1;
				n_prev = init_point2D(n_curr.x,n_curr.y);
				s_prev = (order2>0 ? s_curr/2. : s_curr);
			}
			else{
				n_prev = init_point2D(n_curr.x,n_curr.y);
				s_prev = s_curr;
				j++;
			}
		}
	}
}

void print_poly_on_mesh(char* name,int* Poly,int size){
	int i;
	double s;
	double* Res = zero_vector(glob_mesh.size);
	for (i=0;i<size;i++){
		s = (double)i/(size-1)/2;
		Res[Poly[i]] = 0.5+s;
	}
	print_scalar_data(name,Res,glob_mesh.size);
	free(Res);
}

void print_area_weighted_vector(char* Name,double* Data){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* X = clone_vector(Data,n);
	vector_pseudo_div(X,V,n);
	print_scalar_data(Name,X,n);
	
	free(V);
	free(X);
}

static void remove_duplicates(point2D*** X,int* size,double tol){
	int i,j;
	point2D** CP = (point2D**)malloc((*size)*sizeof(point2D*));
	memcpy(CP,*X,(*size)*sizeof(point2D*));
	for (i=0;i<(*size);i++){
		for (j=0;j<i;j++) if (dist(CP[i],CP[j])<tol) (*X)[i] = NULL;
	}
	for (i=0;i<(*size);i++) if ((*X)[i]==NULL) free(CP[i]);
	
	i = (*size)-1;
	while(i>=0){
		if ((*X)[i]==NULL){
			for (j=i;j<(*size)-1;j++) (*X)[j] = (*X)[j+1];
			(*size)--;
		}
		i--;
	}
	*X = realloc(*X,(*size)*sizeof(point2D*));
	free(CP);
}

int in_polygon(int* Poly,int size,point2D* P){
	const double tol = 1e-10;
	
	int i,j;
	double d;
	point2D* Q1;
	point2D* Q2;
	point2D* I;
	
	// compute polygon center of mass 
	point2D com = init_point2D(0,0);
	for (i=0;i<size;i++) vec_add_mult(&com,&(glob_mesh.Points[Poly[i]]),1.);
	vec_mult(&com,(double)1/size);
	
	// compute distance to most far point
	double max = 0;	
	for (i=0;i<size;i++){
		d = sqrdist(&(glob_mesh.Points[Poly[i]]),&com);
		if (d>max) max = d;
	}
	
	// add 3/2*distance to com -> outer
	point2D ex = init_point2D(1.,0);
	point2D* Outer = clone_point(&com);
	vec_add_mult(Outer,&ex,1.5*sqrt(max));
	point2D** X = (point2D**)malloc(size*sizeof(point2D*));
	
	int count = 0;
	for (i=0;i<size;i++){
		j = (i<size-1) ? i+1 : 0;
		Q1 = &(glob_mesh.Points[Poly[i]]);
		Q2 = &(glob_mesh.Points[Poly[j]]);
		I = line_intersection_positive_2D(P,Outer,Q1,Q2,tol);
		if (I!=NULL){
			X[count] = I;
			count++;
		}		
	}
	
	// remove duplicate intersections
	remove_duplicates(&X,&count,tol);
	
	//clean
	for (i=0;i<count;i++) free(X[i]);
	free(X);
	free(Outer);
	
	return (count % 2);
}

static int* map_upper_triangular(int** Matrix,int dim){	
	
	// Attention: not finished yet
	// should return an index map, that makes M a triangular matrix
	// map[i] is the value index for polygon i 
	
	return get_ID_index_map(dim);
}

double* spread_values2D(mesh2D* Mesh,int** Polygons,int* Sizes,point2D* Seeds,double* Values,int poly_num,int attr_num){
	
	int cmp(void* X1,void* X2){
		int* x1 = (int*)X1;
		int* x2 = (int*)X2;
		if ((*x1)>(*x2)) return 1; else return 0;
	}
	
	int i,j,k,l,m,pos,na,seed,flag,next_size,index;									
	
	int n = Mesh->size;
	double* Res = zero_vector(attr_num*n);
	int* Set = zero_int_list(n);
	
	k = 0;
	int* Polys = NULL;
	for (i=0;i<poly_num;i++){
		k += Sizes[i];
		Polys = (int*)realloc(Polys,k*sizeof(int));
		for (j=0;j<Sizes[i];j++) Polys[k-Sizes[i]+j] = Polygons[i][j];
	}
	int total_size = k;
	
	int** Matrix = (int**)malloc(poly_num*sizeof(int*));
	for (i=0;i<poly_num;i++) Matrix[i] = zero_int_list(poly_num);
	for (i=0;i<poly_num;i++){
		for (k=0;k<poly_num;k++) if (in_polygon(Polygons[i],Sizes[i],&(Seeds[k]))) Matrix[i][k] = 1;
	}
	int* Map = map_upper_triangular(Matrix,poly_num);											// Attention: not finished !
	for (i=0;i<poly_num;i++){
		m = Map[i];
		for (j=0;j<Sizes[i];j++){
			for (l=0;l<attr_num;l++) Res[Polygons[i][j]+l*n] = Values[m+l*poly_num];				
			Set[Polygons[i][j]] = 1;
		}
	}
	for (i=0;i<poly_num;i++) free(Matrix[i]);
	free(Matrix);
	free(Map);
	
	int_sort(Polys,total_size);
	for (i=0;i<poly_num;i++){
		int* Next = NULL;	
		seed = Get_nearest_index(Mesh,&(Seeds[i]));
		Set[seed] = 1;
		for (l=0;l<attr_num;l++) Res[seed+l*n] = Values[i+l*poly_num];
		int bound_size = 1;
		int* Bound = zero_int_list(bound_size);
		Bound[0] = seed;
		
		while(bound_size>0){
			Next = NULL;
			next_size = 0;
			//printf("region %d: %d\n",i,bound_size);
			for(j=0;j<bound_size;j++){
				index = Bound[j];
				for (k=0;k<Mesh->Sizes[index];k++){
					na = Mesh->Connections[index][k];					
					pos = get_position(Polys,total_size,na,&flag);
					if (!Set[na] && flag==-1){
						for (l=0;l<attr_num;l++) Res[na+l*n] = Values[i+l*poly_num];
						Set[na] = 1;						
						next_size++;
						Next = (int*)realloc(Next,next_size*sizeof(int));
						Next[next_size-1] = na;						
					}
				}
			}
			free(Bound);
			Bound = Next;
			bound_size = next_size;
		}
		//printf("fertig !\n\n");		
	}

	free(Set);
	return Res;
}

void boundary_degree_two_case(int i,int j,int k,point2D* Ds){
	int* List;
	point2D ds;
	List = overlap(i,j);
	if (List[1]>=0){
		ds = bound_line_2D(i,j,k);
	}
	else{
		free(List);
		List = overlap(k,i);
		if (List[1]>=0){
			ds = bound_line_2D(k,i,j);
		}
		else{
			free(List);
			List = overlap(j,k);
			if (List[1]>=0){
				ds = bound_line_2D(j,k,i);			
			}			
		}
	}
	free(List);
	Ds->x = -ds.x;
	Ds->y = -ds.y;
}

double max_edge_len(){
	int i,j;
	double l;
	double lmax = dist(&glob_mesh.Points[0],&glob_mesh.Points[glob_mesh.Connections[0][0]]);
	for (i=0;i<glob_mesh.size;i++){
		for (j=0;j<glob_mesh.Sizes[i];j++){
			l = dist(&glob_mesh.Points[i],&glob_mesh.Points[glob_mesh.Connections[i][j]]);
			if (l>lmax) lmax = l;
		}
	}
	return lmax;
}

double mean_edge_len(){
	int i,j;
	double l;
	double mean = 0;
	for (i=0;i<glob_mesh.size;i++){
		l = 0;
		for (j=0;j<glob_mesh.Sizes[i];j++){
			l += dist(&glob_mesh.Points[i],&glob_mesh.Points[glob_mesh.Connections[i][j]]);
		}
		l /= (double)glob_mesh.Sizes[i];
		mean += l;
	}
	return (double)mean/glob_mesh.size;
}

/*! 
 *  @param i: Knotennummer 
 *  @return Gibt den Listenindex einer Randverbindung falls ein Randpunkt vorliegt,
 * sonst -1. 
*/

int is_boundary(int i){																// no boundary returns -1, else return >=0
	int j;
	int* list;
	int res = -1;
	for (j=0;j<glob_mesh.Sizes[i];j++){
		list = overlap(i,glob_mesh.Connections[i][j]);
		if (list[1]<0){
			res = j;
			free(list);
			break;
		}else{free(list);}
	}
	return res;
}

int IsBoundary(mesh2D* Mesh,int i){																// no boundary returns -1, else return >=0
	int j;
	int* list;
	int res = -1;
	for (j=0;j<Mesh->Sizes[i];j++){
		list = Overlap(Mesh,i,Mesh->Connections[i][j]);
		if (list[1]<0){
			res = j;
			free(list);
			break;
		}else{free(list);}
	}
	return res;
}

int IsBoundaryEdge(mesh2D* Mesh,index2D ind){	
	int res = 0;
	int* list = Overlap(Mesh,ind.i,ind.j);
	if (list[1]==-1) res = 1;
	free(list);
	return res;
}

int near_boundary(int i){															// near boundary returns 1, else 0
	int j,k;
	int* list;	
	if (is_boundary(i)>=0) return 0;
	for (j=0;j<glob_mesh.Sizes[i];j++){
		if (j==0) k = glob_mesh.Sizes[i]; else k = j;		
		list = overlap(glob_mesh.Connections[i][j],glob_mesh.Connections[i][k-1]);
		if (list[1]<0){
			free(list);
			return 1;
		}
		free(list);
	}
	return 0;
}

int get_boundary_path(int start,int end,int init_prev,int**List,int* len){
	int prev,curr,next;
	int max_len = glob_mesh.size;
	*List = (int*)malloc(sizeof(int));
	*len = 1;
	(*List)[0] = start;
	prev = init_prev;	
	curr = start;
	do{
		next = glob_mesh.Connections[curr][0];
		if (next==prev) next = glob_mesh.Connections[curr][glob_mesh.Sizes[curr]-1];
		(*len)++;
		*List = (int*)realloc(*List,(*len)*sizeof(int));
		(*List)[(*len)-1] = next;
		prev = curr;
		curr = next;
	}while(curr!=end && (*len)<max_len);
	if ((*len)>=max_len) return 0; else return 1;
}

void get_shortest_boundary_path(int start,int end,int** List,int* len){
	int* List1;
	int* List2;
	int len1,len2,success1,success2;
	int prev1 = glob_mesh.Connections[start][0];
	int prev2 = glob_mesh.Connections[start][glob_mesh.Sizes[start]-1];
	success1 = get_boundary_path(start,end,prev1,&List1,&len1);
	success2 = get_boundary_path(start,end,prev2,&List2,&len2);
	if (!success1 && !success2){
		printf("get_boundary_path: failed\n");
		return;
	}
	if (success1 && !success2){
		free(List2);
		*len = len1;
		*List = List1;
		return;
	}
	if (!success1 && success2){
		free(List1);
		*len = len2;
		*List = List2;
		return;
	}
	if (success1 && success2){
		if (len1>len2){
			free(List1);
			*len = len2;
			*List = List2;
		}
		else{
			free(List2);
			*len = len1;
			*List = List1;
		}
	}
}

int point_met(point2D* List,int size,point2D* P,double tol){
	int i;
	
	int res = -1;
	for (i=0;i<size;i++) if (sqrdist(&(List[i]),P)<tol){
		res = i;
		break;
	}
	return res;
}

int get_orient_dir(int start,point2D* ChangeB,int sizeB,double tol){
	
	int ChPoint,next;
	
	int prev = -1;
	int current = start;
	
	if (sizeB==1) return 1;
	
	do{			
		
		next = (glob_mesh.Connections[current][0]!=prev) ? glob_mesh.Connections[current][0] : glob_mesh.Connections[current][glob_mesh.Sizes[current]-1];
		if (is_boundary(next)>=0){			
			ChPoint = point_met(ChangeB,sizeB,&(glob_mesh.Points[next]),tol);
			if (ChPoint>=0){
				if (ChPoint==1) return 1;
				if (ChPoint==sizeB-1) return -1;								
			}
			prev = current;
			current = next;
		}
		else{
			printf("Warning: boundary ended abruptly at index %d\n",current);
			break;
		}
	}while(current!=start);	
	
	return 0;
}
	

int* get_boundary_conditions(point2D** ChangeB,int** Cond,int* Sizes,int loops){		// result is vector of size 3*n
	const double tol = 1e-20;															// 1. Condition
	const double dmax = 0.01;															// 2. loop index i
	int i,j,k,c,next,current,prev,dir;													// 3. segment index j
	double d,min;
	
	int n = glob_mesh.size;
	int* Res = zero_int_list(3*n);
	for (i=0;i<loops;i++){
		int start = -1;
		min = 1./tol;
		for (k=0;k<n;k++) if (is_boundary(k)>=0){
			d = sqrdist(&(glob_mesh.Points[k]),&(ChangeB[i][0]));
			if (d<min){
				min = d;
				start = k;
			}
		}
		//printf("index: %d, sqr=%e\n",start,min);
		if (min>dmax){
			printf("No boundary path in range %f of point %d\n",sqrt(dmax),i);
			free(Res);
			return NULL;
		}
		
		dir = get_orient_dir(start,ChangeB[i],Sizes[i],tol);
		if (dir==0){
			printf("Bad condition file: something's wrong with point order -> abort\n");
			return NULL;
		}
		
		j = 0;//(dir==-1) ? Sizes[i]-1 : 0;
		prev = -1;
		current = start;	
		
		do{		
			c = (dir==-1) ? j-1 : j;	
			if (c<0) c = Sizes[i]-1;
			Res[current] = Cond[i][c];
			Res[n+current] = i;
			Res[2*n+current] = c;
			next = (glob_mesh.Connections[current][0]!=prev) ? glob_mesh.Connections[current][0] : glob_mesh.Connections[current][glob_mesh.Sizes[current]-1];
			if (is_boundary(next)>=0){		
				k = j+dir;
				if (k<0) k = Sizes[i]-1;
				if (k>=Sizes[i]) k = 0;						
				if (sqrdist(&(glob_mesh.Points[next]),&(ChangeB[i][k]))<tol){ 
					if (dir==-1 && k>0) j = k;
					if (dir==1 && k<Sizes[i]) j = k;				
				}
				prev = current;
				current = next;
			}
			else{
				printf("Warning: boundary ended abruptly at index %d\n",current);
				break;
			}
		}while(current!=start);						
	}
	return Res;
}
	

/*int boundary_edge(int i,int j,int k,int** B_list,int** I_list){
	int len = 0;
	int ilen = 0;
	*B_list = NULL;
	*I_list = NULL;
	if (is_boundary(i)>=0){
		len++;
		*B_list = (int*)realloc(*B_list,len*sizeof(int));
		(*B_list)[len-1] = i;
	}
	else{
		ilen++;
		*I_list = (int*)realloc(*I_list,ilen*sizeof(int));
		(*I_list)[ilen-1] = i;
	}
	
	if (is_boundary(j)>=0){
		len++;
		*B_list = (int*)realloc(*B_list,len*sizeof(int));
		(*B_list)[len-1] = j;
	}
	else{
		ilen++;
		*I_list = (int*)realloc(*I_list,ilen*sizeof(int));
		(*I_list)[ilen-1] = j;
	}
	
	if (is_boundary(k)>=0){
		len++;
		*B_list = (int*)realloc(*B_list,len*sizeof(int));
		(*B_list)[len-1] = k;
	}
	else{
		ilen++;
		*I_list = (int*)realloc(*I_list,ilen*sizeof(int));
		(*I_list)[ilen-1] = k;
	}
	
	return len;
}*/

/*! 
 *  @param keine
 *  @return sortiert die Nachbarknoten in geometrischer Reihenfolge
*/

void sort_knots(){
	int i;
	for (i=0;i<glob_mesh.size;i++){
		glob_mesh.Is_boundary[i] = is_boundary(i);
	}
	for (i=0;i<glob_mesh.size;i++){
		sortknot(i);
	}
}

/*! 
 *  @param i:Knotennummer
 *  @return sortiert die Nachbarknoten in geometrischer Reihenfolge
*/

void sortknot(int i){
	int j,next,last,start,h;
	int len = glob_mesh.Sizes[i];
	int* buffer = (int*)malloc(len*sizeof(int));
	int* list;
	if (glob_mesh.Is_boundary[i]<0){
		last = glob_mesh.Connections[i][0];
		list = overlap(i,last);
		start = list[0];
		next = list[1];
		free(list);
		buffer[0] = last;
		j = 0;
		do {
			j++;
			h = next;
			list = overlap(i,next);
			if (list[0]!=last){next = list[0];}
			else{next = list[1];}
			last = h;
			buffer[j] = last;
			free(list);
		}while(next!=start);
		j++;
		buffer[j] = start;
	}
	else{
		last = glob_mesh.Connections[i][glob_mesh.Is_boundary[i]];
		buffer[0] = last;
		list = overlap(i,last);
		next = list[0];
		free(list);
		j = 0;
		do {
			j++;
			h = next;
			list = overlap(i,next);
			if (list[0]!=last){next = list[0];}
			else{next = list[1];}
			last = h;
			buffer[j] = last;
			free(list);
		}while(next>=0);
	}
	if (j+1!=glob_mesh.Sizes[i]){printf("Fehler in Knoten %d\n",i);}
	free(glob_mesh.Connections[i]);
	glob_mesh.Connections[i] = buffer;
}

void create_look_up_table(mesh2D* Mesh){
	int i,j,m,k;
	int* list;
	int n = Mesh->size;
	if (Mesh->Overlap_table==NULL){
		Mesh->Overlap_table = (int***)malloc(n*sizeof(int**));
		for (i=0;i<n;i++){
			m = Mesh->Sizes[i];
			Mesh->Overlap_table[i] = (int**)malloc(m*sizeof(int*));
			for (j=0;j<m;j++){
				k = Mesh->Connections[i][j];
				list = init_overlap(Mesh,i,k);
				Mesh->Overlap_table[i][j] = (int*)malloc(3*sizeof(int));
				Mesh->Overlap_table[i][j][0] = k;
				Mesh->Overlap_table[i][j][1] = list[0];
				Mesh->Overlap_table[i][j][2] = list[1];
				free(list);
			}
		}
	}
}

int* Overlap(mesh2D* Mesh,int i,int j){
	int k,l,n;
	int counter = 0;
	int* Res = (int*)malloc(2*sizeof(int));
	if (Mesh->Overlap_table==NULL){
		Res[0] = -1;
		Res[1] = -1;
		for (k=0;k<Mesh->Sizes[i];k++){
			n = Mesh->Connections[i][k];
			for (l=0;l<Mesh->Sizes[j];l++){
				if  (Mesh->Connections[j][l]==n){
					Res[counter] = n;
					counter++;
				}
			}
		}
		if (counter > 2 || counter==0){
			printf("Connection error in node %d, size: %d\n",i,Mesh->size);
			if (Mesh->Sizes[i]==0) printf("Maybe a duplicate node was generated\n");
		}
		return Res;
	}
	else{
		for (k=0;k<Mesh->Sizes[i];k++){
			if (Mesh->Overlap_table[i][k][0]==j){
				Res[0] = Mesh->Overlap_table[i][k][1];
				Res[1] = Mesh->Overlap_table[i][k][2];
				return Res;
			}
		}
		Res[0] = -1;
		Res[1] = -1;
		printf("Error in look-up table at node %d\n",i);
		return Res;
	}
}

void Sortknot(mesh2D* Mesh,int i){
	int j,next,last,start,h;
	int len = Mesh->Sizes[i];
	int* buffer = (int*)malloc(len*sizeof(int));
	int* list;
	if (Mesh->Is_boundary[i]<0){
		last = Mesh->Connections[i][0];
		list = Overlap(Mesh,i,last);
		start = list[0];
		next = list[1];
		free(list);
		buffer[0] = last;
		j = 0;
		do {
			j++;
			h = next;
			list = Overlap(Mesh,i,next);
			if (list[0]!=last){next = list[0];}
			else{next = list[1];}
			last = h;
			buffer[j] = last;
			free(list);
		}while(next!=start);
		j++;
		buffer[j] = start;
	}
	else{
		last = Mesh->Connections[i][Mesh->Is_boundary[i]];
		buffer[0] = last;
		list = Overlap(Mesh,i,last);
		next = list[0];
		free(list);
		j = 0;
		do {
			j++;
			h = next;
			list = Overlap(Mesh,i,next);
			if (list[0]!=last){next = list[0];}
			else{next = list[1];}
			last = h;
			buffer[j] = last;
			free(list);
		}while(next>=0);
	}
	if (j+1!=Mesh->Sizes[i]){printf("Fehler in Knoten %d\n",i);}
	free(Mesh->Connections[i]);
	Mesh->Connections[i] = buffer;
}


void Sort_knots(mesh2D* Mesh){
	int i;
	for (i=0;i<Mesh->size;i++){
		Mesh->Is_boundary[i] = IsBoundary(Mesh,i);
	}
	for (i=0;i<Mesh->size;i++){
		Sortknot(Mesh,i);
	}
}

point2D get_voronoi_edge(point2D* P2,point2D* P3){
	double x2,x3,y2,y3,s;
	point2D res;
	x2 = P2->x;
	y2 = P2->y;
	x3 = P3->x;
	y3 = P3->y;
	s = -2*x3*y2+2*x2*y3;
	res.x = -x3*x3*y2+(x2*x2+y2*(y2-y3))*y3;
	res.y = -x3*y2*y2+(y3*y3+x3*(x3-x2))*x2;
	vec_mult(&res,(double)1/s);
	return res;
}

point2D* get_voronoi_edges(int k,double* Shift){
	int i,k2,k3,s;  // hier läuft was falsch !
	point2D* P2;
	point2D* P3 = NULL;
	int d = glob_mesh.size;
	int n = glob_mesh.Sizes[k];
	int bound = is_boundary(k);
	if (bound>=0) s = n-1; else s = n;
	point2D* Res = (point2D*)malloc(s*sizeof(point2D));
	point2D* P1 = clone_point(&glob_mesh.Points[k]);
	P1->x += Shift[k];
	P1->y += Shift[k+d];
	for (i=0;i<n;i++){
		k2 = glob_mesh.Connections[k][i];
		if (bound>=0){
			if (i<n-1) k3 = glob_mesh.Connections[k][i+1];
			else break;
		}
		else {
			if (i<n-1) k3 = glob_mesh.Connections[k][i+1];
			else k3 = glob_mesh.Connections[k][0];
		}
		if (i==0) {
			P2 = clone_point(&glob_mesh.Points[k2]);
			P2->x += Shift[k2]-P1->x;
			P2->y += Shift[k2+d]-P1->y;
		}
		else P2 = P3;
		P3 = clone_point(&glob_mesh.Points[k3]);
		P3->x += Shift[k3]-P1->x;
		P3->y += Shift[k3+d]-P1->y;
		Res[i] = get_voronoi_edge(P2,P3);
		free(P2);
	}
	free(P3);
	free(P1);
	return Res;
}

double voronoi_area(point2D* P2,point2D* P3){
	double x2,x3,y2,y3,s,a2,a3;
	x2 = P2->x;
	y2 = P2->y;
	x3 = P3->x;
	y3 = P3->y;
	s = fabs((double)8*(-x3*y2+x2*y3));
	a2 = fabs((x2*x2+y2*y2)*(x3*(x3-x2)+y3*(y3-y2)));
	a3 = fabs((x3*x3+y3*y3)*(x2*(x2-x3)+y2*(y2-y3)));
	return (a2+a3)/s;
}

double get_voronoi_area(int k,double* Shift){
	int i,k2,k3;
	point2D* P2;
	point2D* P3 = NULL;
	int d = glob_mesh.size;
	int n = glob_mesh.Sizes[k];
	int bound = is_boundary(k);
	point2D* P1 = clone_point(&glob_mesh.Points[k]);
	P1->x += Shift[k];
	P1->y += Shift[k+d];
	double res = 0;
	for (i=0;i<n;i++){
		k2 = glob_mesh.Connections[k][i];
		if (bound>=0){
			if (i<n-1) k3 = glob_mesh.Connections[k][i+1];
			else break;
		}
		else {
			if (i<n-1) k3 = glob_mesh.Connections[k][i+1];
			else k3 = glob_mesh.Connections[k][0];
		}
		if (i==0) {
			P2 = clone_point(&glob_mesh.Points[k2]);
			P2->x += Shift[k2]-P1->x;
			P2->y += Shift[k2+d]-P1->y;
		}
		else P2 = P3;
		P3 = clone_point(&glob_mesh.Points[k3]);
		P3->x += Shift[k3]-P1->x;
		P3->y += Shift[k3+d]-P1->y;
		res += voronoi_area(P2,P3);
		free(P2);
	}
	free(P3);
	free(P1);
	return res;
}

point2D* get_flux_coefficients(int k,double* Shift){
	int i,j;   // oder hier ?
	double s;
	point2D* E2;
	point2D* E3 = NULL;
	point2D* N;
	int d = glob_mesh.size;
	int n = glob_mesh.Sizes[k];
	int bound = is_boundary(k);
	double a = get_voronoi_area(k,Shift);
	point2D* Edges = get_voronoi_edges(k,Shift);
	point2D* P1 = clone_point(&glob_mesh.Points[k]);
	P1->x += Shift[k];
	P1->y += Shift[k+d];
	point2D* Res = (point2D*)malloc(n*sizeof(point2D));
	for (i=0;i<n;i++){
		j = glob_mesh.Connections[k][i];
		N = clone_point(&glob_mesh.Points[j]);
		N->x += Shift[j];
		N->y += Shift[j+d];
		if (i==0) {
			E2 = &Edges[n-1]; 
		}
		else {
			E2 = &Edges[i-1];
			E3 = &Edges[i];
		}
		if (bound>=0){
			if (i==0){
				E2 = N;
				E2->x -= P1->x;
				E2->y -= P1->y;
				vec_mult(E2,0.5);
				E3 = &Edges[i];
			}
			if (i==n-1){
				E3 = N;
				E3->x -= P1->x;
				E3->y -= P1->y;
				vec_mult(E3,0.5);
			}
		}
		s = dist(E2,E3)/a;
		normalize(N);
		vec_mult(N,s);
		Res[i] = *N;
		free(N);
	}
	free(P1);
	free(Edges);
	return Res;
}

sparse_matrix* stretch_matrix(double ax,double ay,int var_num,int var_ind){
	int i;
	int d = glob_mesh.size;
	sparse_matrix* Trans = sparse_identity(var_num*d);
	for (i=0;i<d;i++){
		Trans->Values[var_ind*d+i][0] *= ax;
		Trans->Values[(var_ind+1)*d+i][0] *= ay;
	} 
	return Trans;
}

sparse_matrix* get_diag_matrix_function(int var_num,double (*F)(int mesh_ind,int var_ind)){
	int i,j;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(d*var_num);
	for (i=0;i<var_num;i++){
		for (j=0;j<d;j++) Res->Values[i*d+j][0] = (*F)(j,i);
	}
	return Res;
}

point2D* center2D(int i,int j,int k){
	double fac = (double)1/3;
	point2D* C = clone_point(&glob_mesh.Points[i]);
	vec_mult(C,fac);
	vec_add_mult(C,&glob_mesh.Points[j],fac);
	vec_add_mult(C,&glob_mesh.Points[k],fac);
	return C;
}

point2D* get_dual_knot(int k){
	int i,k2,k3;
	point2D* C;
	int n = glob_mesh.Sizes[k];
	point2D* Duals = (point2D*)malloc(n*sizeof(point2D));
	k2 = glob_mesh.Connections[k][n-1];
	for (i=0;i<n;i++){
		k3 = glob_mesh.Connections[k][i];
		C = center2D(k,k2,k3);
		Duals[i] = *C;
		k2 = k3;
		free(C);
	}
	return Duals;
}

point2D* get_dual_bound_knot(int k){
	int i,k2,k3;
	point2D* C;
	int n = glob_mesh.Sizes[k];
	point2D* Duals = (point2D*)malloc((n-1)*sizeof(point2D));
	k2 = glob_mesh.Connections[k][0];
	for (i=1;i<n;i++){
		k3 = glob_mesh.Connections[k][i];
		C = center2D(k,k2,k3);
		Duals[i-1] = *C;
		k2 = k3;
		free(C);
	}
	return Duals;
}

point2D* get_flux_weights(int k){
	int i,j;
	int n = glob_mesh.Sizes[k];
	double a,d;
	point2D* Dual = get_dual_knot(k);
	point2D* Lines = (point2D*)malloc(n*sizeof(point2D));
	a = polygon_area(Dual,n);
	j = 1;
	for (i=0;i<n;i++){
		d = dist(&Dual[j],&Dual[i]);
		Lines[i] = get_normal(&Dual[j],&Dual[i],&glob_mesh.Points[k]);
		vec_mult(&Lines[i],d/a);
		if (i<n-2) j++; else j = 0;
	}
	free(Dual);
	return Lines;
}

point2D* get_flux_bound_weights(int k){
	int i,j;
	int n = glob_mesh.Sizes[k]-1;
	double a,d;
	point2D* P;
	point2D* Q;
	point2D* Dual = (point2D*)get_dual_bound_knot(k);
	Dual = (point2D*)realloc(Dual,(n+3)*sizeof(point2D));
	Q = &glob_mesh.Points[k];
	P = &glob_mesh.Points[glob_mesh.Connections[k][n-1]];
	Dual[n].x = (P->x+Q->x)/2.;
	Dual[n].y = (P->y+Q->y)/2.;
	Dual[n+1].x = Q->x;
	Dual[n+1].y = Q->y;
	P = &glob_mesh.Points[glob_mesh.Connections[k][0]];
	Dual[n+2].x = (P->x+Q->x)/2.;
	Dual[n+2].y = (P->y+Q->y)/2.;
	a = polygon_area(Dual,n);
	point2D* Lines = (point2D*)malloc((n+1)*sizeof(point2D));
	for (i=0;i<n+1;i++){
		if (i>0) j = i-1; else j = n+2;
		d = dist(&Dual[i],&Dual[j]);
		Lines[i] = get_normal(&Dual[i],&Dual[j],&glob_mesh.Points[k]);
		vec_mult(&Lines[i],d/a);
	}
	free(Dual);
	return Lines;
}

point2D* flux_weights(int k){					// Größe des Arrays ist Größe des Knotens
	point2D* Res;
	if (is_boundary(k)>=0) Res = get_flux_bound_weights(k);
	else Res = get_flux_weights(k);
	return Res;
}

double val_weight(int i,int j){
	double a,b;
	int* List = overlap(i,j);
	if (List[1]<0){
		free(List);
		return 0.5;
	}
	else{
		point2D* Q1 = &glob_mesh.Points[i];
		point2D* Q2 = &glob_mesh.Points[j];
		point2D* P1 = &glob_mesh.Points[List[0]];
		point2D* P2 = &glob_mesh.Points[List[1]];
		a = P1->y*(-P2->x+Q1->x)+P1->x*(P2->y-Q1->y)+P2->x*Q1->y-P2->y*Q1->x;
		b = (P1->y-P2->y)*(Q1->x-Q2->x)-(P1->x-P2->x)*(Q1->y-Q2->y);
		free(List);
		return a/b;
	}
}

double* val_weights(int k){
	int i,j;
	int n = glob_mesh.Sizes[k];
	double* Res = (double*)malloc(n*sizeof(double));
	for (i=0;i<n;i++) {
		j = glob_mesh.Connections[k][i];
		Res[i] = val_weight(k,j);
		/*if (Res[i]<0){
			printf("Fehler bei FVM-Wertegewichtung bei Index %d|%d Wert: %f\n",k,j,Res[i]);
		}*/
		if (Res[i]<0) Res[i] = 0;
		if (Res[i]>1) Res[i] = 1;
	}
	return Res;
}

void test_dual(){
	int i,j,len;
	double L;
	point2D sum;
	int n = glob_mesh.size;
	for (i=0;i<n;i++){
		sum.x = 0;
		sum.y = 0;
		L = 0;
		len = glob_mesh.Sizes[i];
		for (j=0;j<len;j++){
			vec_add_mult(&sum,&(Dual_mesh[i][j]),1.);
			L += vec_abs(&(Dual_mesh[i][j]));
		}
		if (is_boundary(i)>=0){
			vec_add_mult(&sum,&(Dual_mesh[i][len]),1.);
			vec_add_mult(&sum,&(Dual_mesh[i][len+1]),1.);
			L += vec_abs(&(Dual_mesh[i][len]));
			L += vec_abs(&(Dual_mesh[i][len+1]));
		}
		L = vec_abs(&sum)/L;
		if (L>1E-3) printf("sum: %f at index %d\n",L,i);
	}
}

sparse_matrix* FVM_convection_matrix2D(double* Flow_field,double* Shift_field,sparse_matrix* Coeff,double dt){
	int i,j,k,len;
	double sum,s,a;
	point2D J;
	int d = glob_mesh.size;
	sparse_matrix* A = sparse_zero(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		a = dt/(4.*Dual_areas[i]);
		sum = 0;
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			J.x = Flow_field[i]+Flow_field[k]-Shift_field[i]-Shift_field[k];
			J.y = Flow_field[i+d]+Flow_field[k+d]-Shift_field[i+d]-Shift_field[k+d];
			s = vec_scalar(&J,&Dual_mesh[i][j]);
			sum += s;
			insert_sparse(A,s*a,i,k);
			if (Coeff!=NULL) insert_sparse(Coeff,s*a/dt,i,k);
		}
		s = 0;
		if (is_boundary(i)>=0){
			J.x = Flow_field[i];
			J.y = Flow_field[i+d];
			s = vec_scalar(&Dual_mesh[i][len],&J)+vec_scalar(&Dual_mesh[i][len+1],&J);
		}
		insert_sparse(A,1.+a*sum+4.*a*s,i,i);
		if (Coeff!=NULL) insert_sparse(Coeff,(a*sum+4.*a*s)/dt,i,i);
	}
	return A;
}

/*sparse_matrix* FVM_convective_fluxes_2D(double* Density,double* Flow_field,double* Shift_field){
	int i,j,k,len;
	double s,a,c;
	point2D J;
	int d = glob_mesh.size;
	sparse_matrix* A = sparse_zero(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		a = 1./(Dual_areas[i]);
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			if (Density[i]==0 && Density[k]==0) c = 0; else c = Density[i]/(Density[i]+Density[k]);
			J.x = Flow_field[i]+Flow_field[k]-Shift_field[i]-Shift_field[k];
			J.y = Flow_field[i+d]+Flow_field[k+d]-Shift_field[i+d]-Shift_field[k+d];
			s = vec_scalar(&J,&Dual_mesh[i][j]);
			insert_sparse(A,-s*a*c,i,k);
		}
		s = 0;
		if (is_boundary(i)>=0){
			J.x = Flow_field[i]-Shift_field[i];
			J.y = Flow_field[i+d]-Shift_field[i+d];
			s = vec_scalar(&Dual_mesh[i][len],&J)+vec_scalar(&Dual_mesh[i][len+1],&J);
		}
		insert_sparse(A,-s*a,i,i);
	}
	return A;
}*/

/*sparse_matrix* FVM_convective_fluxes_2D(double* Density,double* Flow_field,double* Shift_field){
	int i,j,k,len;
	double s,a,c;
	point2D J;
	int d = glob_mesh.size;
	sparse_matrix* A = sparse_zero(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		a = 1./(Dual_areas[i]);
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			if (Density[i]==0 && Density[k]==0) c = 0; else c = 0.5/(Density[i]+Density[k]);
			J.x = Flow_field[i]+Flow_field[k]-Shift_field[i]-Shift_field[k];
			J.y = Flow_field[i+d]+Flow_field[k+d]-Shift_field[i+d]-Shift_field[k+d];
			s = vec_scalar(&J,&Dual_mesh[i][j]);
			insert_sparse(A,-s*a*c*Density[i],i,k);
			insert_sparse(A,-s*a*c*Density[k],k,i);
			// symmetrisch um Massenerhaltung zu garantieren
		}
		s = 0;
		if (is_boundary(i)>=0){
			J.x = Flow_field[i]-Shift_field[i];
			J.y = Flow_field[i+d]-Shift_field[i+d];
			s = vec_scalar(&Dual_mesh[i][len],&J)+vec_scalar(&Dual_mesh[i][len+1],&J);
		}
		insert_sparse(A,-s*a,i,i);
	}
	return A;
}*/

sparse_matrix* FVM_convective_fluxes_2D(double* Density,double* Flow_field,double* Shift_field){
	int i,j,k,len;
	double s,a,ci,ck;
	point2D J;
	int d = glob_mesh.size;
	double* U = set_vector_Ui_0_2D(1);
	sparse_matrix* A = sparse_zero(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		//a = 1./(Dual_areas[i]);
		a = 1./U[i];
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			if (Density[i]==0 && Density[k]==0){
				ci = 0;
				ck = 0;
			}
			else{
				ci = Density[i]/(Density[i]+Density[k]);
				ck = Density[k]/(Density[i]+Density[k]);
				if (partition_mode==PARTITION_ON){
					ci *= Partition_attributes[max(Partition[i],Partition[k])-1];
					ck *= Partition_attributes[max(Partition[i],Partition[k])-1];
				}
			}
			J.x = 0.5*(Flow_field[i]+Flow_field[k]-Shift_field[i]-Shift_field[k]);
			J.y = 0.5*(Flow_field[i+d]+Flow_field[k+d]-Shift_field[i+d]-Shift_field[k+d]);
			s = vec_scalar(&J,&Dual_mesh[i][j]);
			insert_sparse(A,-s*a*ci,i,k);
			insert_sparse(A,-s*a*ck,i,i);
		}
		s = 0;
		if (is_boundary(i)>=0){
			J.x = Flow_field[i]-Shift_field[i];
			J.y = Flow_field[i+d]-Shift_field[i+d];
			s = vec_scalar(&Dual_mesh[i][len],&J)+vec_scalar(&Dual_mesh[i][len+1],&J);
			if (partition_mode==PARTITION_ON){
				s *= Partition_attributes[max(Partition[i],Partition[k])-1];
			}
		}
		insert_sparse(A,-s*a,i,i);
	}
	free(U);
	return A;
}

/*sparse_matrix* FVM_convective_fluxes_2D(double* Density,double* Flow_field,double* Shift_field){
	int i,j,k,len;
	double s,a,c;
	point2D J;
	int d = glob_mesh.size;
	sparse_matrix* A = sparse_zero(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		a = 1./(Dual_areas[i]);
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			if (Density[i]==0 && Density[k]==0) c = 0; else c = Density[i]/(Density[i]+Density[k]);
			J.x = Flow_field[i]+Flow_field[k]-Shift_field[i]-Shift_field[k];
			J.y = Flow_field[i+d]+Flow_field[k+d]-Shift_field[i+d]-Shift_field[k+d];
			s = vec_scalar(&J,&Dual_mesh[i][j]);
			insert_sparse(A,-s*a*c,i,k);
		}
		s = 0;
		if (is_boundary(i)>=0){
			J.x = Flow_field[i]-Shift_field[i];
			J.y = Flow_field[i+d]-Shift_field[i+d];
			s = vec_scalar(&Dual_mesh[i][len],&J)+vec_scalar(&Dual_mesh[i][len+1],&J);
		}
		insert_sparse(A,-s*a,i,i);
	}
	return A;
}*/


void FVM_convection_2D(double* Density_field,double* Flow_field,double* Shift_field,double dt){
	int i,j,k,len;
	point2D J;
	int d = glob_mesh.size;
	double* D = zero_vector(d);
	for (i=0;i<d;i++){
		len = glob_mesh.Sizes[i];
		for (j=0;j<len;j++){
			k = glob_mesh.Connections[i][j];
			J.x = (Flow_field[i]+Flow_field[k])*(Density_field[i]+Density_field[k])/4.;
			J.y = (Flow_field[i+d]+Flow_field[k+d])*(Density_field[i]+Density_field[k])/4.;
			D[i] -= vec_scalar(&Dual_mesh[i][j],&J);
		}
		if (is_boundary(i)>=0){
			J.x = Flow_field[i]*Density_field[i];
			J.y = Flow_field[i+d]*Density_field[i];
			D[i] -= vec_scalar(&Dual_mesh[i][len],&J);
			D[i] -= vec_scalar(&Dual_mesh[i][len+1],&J);
		}
		D[i] /= Dual_areas[i];
	}
	vector_add(Density_field,D,dt,d);
	double sum = 0;
	for (i=0;i<d;i++) sum += D[i]*Dual_areas[i];
	free(D);
	printf("total mass: %f\n",sum);
}

/*! 
 * @param P1,P2,P3 Pointer auf Punkte eines Dreiecks. Reihenfolge beachten !
 * @return Fläche des Dreiecks
 */

double det(point2D* P1,point2D* P2,point2D* P3){
	return (P1->x)*((P2->y)-(P3->y))
	+(P2->x)*((P3->y)-(P1->y))
	+(P3->x)*((P1->y)-(P2->y));
}

double det_tri(int i,int j,int k){
	point2D* P1 = &(glob_mesh.Points[i]);
	point2D* P2 = &(glob_mesh.Points[j]);
	point2D* P3 = &(glob_mesh.Points[k]);
	return det(P1,P2,P3);
}


double vector_Ui_0(int i){				// integral über phi_i
	int k,j,l;
	int len = glob_mesh.Sizes[i];
	double res = 0;
	for (k=0;k<len-1;k++){
		j = glob_mesh.Connections[i][k];
		l = glob_mesh.Connections[i][k+1];
		res += fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
		 &glob_mesh.Points[l]));
	}
	if (glob_mesh.Is_boundary[i]<0){
		j = glob_mesh.Connections[i][len-1];
		l = glob_mesh.Connections[i][0];
		res += fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
		 &glob_mesh.Points[l]));
		}
	return (double)res/6.;
}

double get_vector_Ui_0(mesh2D* Mesh,int i){				// integral über phi_i
	int k,j,l;
	int len = Mesh->Sizes[i];
	double res = 0;
	for (k=0;k<len-1;k++){
		j = Mesh->Connections[i][k];
		l = Mesh->Connections[i][k+1];
		res += fabs(det(&Mesh->Points[i],&Mesh->Points[j],
		 &Mesh->Points[l]));
	}
	if (Mesh->Is_boundary[i]<0){
		j = Mesh->Connections[i][len-1];
		l = Mesh->Connections[i][0];
		res += fabs(det(&Mesh->Points[i],&Mesh->Points[j],
		 &Mesh->Points[l]));
		}
	return (double)res/6.;
}

/*! 
 * @param i,j Matrixindizes auf dem Gitter 
 * @return quadratisches Überlappintegral ij ohne Ableitung
 */

double vector_b_0(int i){
	int k,j,l;
	int len = glob_mesh.Sizes[i];
	double res = 0;
	for (k=0;k<len-1;k++){
		j = glob_mesh.Connections[i][k];
		l = glob_mesh.Connections[i][k+1];
		res += fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
		 &glob_mesh.Points[l]));
	}
	if (glob_mesh.Is_boundary[i]<0){
		j = glob_mesh.Connections[i][len-1];
		l = glob_mesh.Connections[i][0];
		res += fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
		 &glob_mesh.Points[l]));
		}
	return (double)res/6;	
}

double matrix_Aij_00(int i,int j){
	int len,k;
	double factor;
	int* list = overlap(i,j);
	if (list[1]<0){len = 1;}else{len = 2;}
	double res = 0;
	if (partition_mode && Partition[i]==0 && Partition[j]==0){
		for (k=0;k<len;k++){
			factor = Partition_attributes[Partition[list[k]]-1];
			res += factor*fabs(det(&glob_mesh.Points[list[k]],&glob_mesh.Points[i],
			 &glob_mesh.Points[j]));
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[max(Partition[i],Partition[j])-1]; else factor = 1.;
		for (k=0;k<len;k++){
			res += factor*fabs(det(&glob_mesh.Points[list[k]],&glob_mesh.Points[i],
			 &glob_mesh.Points[j]));
		}
	}
	free(list);
	return (double)res/24;
}

/*! 
 * @param i Matrixindex auf dem Gitter 
 * @return quadratisches Überlappintegral ii ohne Ableitung 
 */

double matrix_Aii_00(int i){
	int k,j,l;
	double factor;
	int len = glob_mesh.Sizes[i];
	double res = 0;
	if (partition_mode && Partition[i]==0){
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			res += factor*fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
			 &glob_mesh.Points[l]));
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			res += factor*fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
			 &glob_mesh.Points[l]));
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[Partition[i]-1]; else factor = 1.;
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			res += factor*fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
			 &glob_mesh.Points[l]));
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			res += factor*fabs(det(&glob_mesh.Points[i],&glob_mesh.Points[j],
			 &glob_mesh.Points[l]));
		}
	}
	return (double)res/12.;
}

double get_matrix_Aij_00(mesh2D* Mesh,int i,int j){
	int len,k;
	int* list = init_overlap(Mesh,i,j);
	if (list[1]<0){len = 1;}else{len = 2;}
	double res = 0;
	
	for (k=0;k<len;k++){
		res += fabs(det(&(Mesh->Points[list[k]]),&(Mesh->Points[i]),&(Mesh->Points[j])));
	}
	free(list);
	return (double)res/24;
}

double get_matrix_Aii_00(mesh2D* Mesh,int i){
	int k,j,l;
	int len = Mesh->Sizes[i];
	double res = 0;

	for (k=0;k<len-1;k++){
		j = Mesh->Connections[i][k];
		l = Mesh->Connections[i][k+1];
		res += fabs(det(&(Mesh->Points[i]),&(Mesh->Points[j]),&(Mesh->Points[l])));
	}
	if (Mesh->Is_boundary[i]<0){
		j = Mesh->Connections[i][len-1];
		l = Mesh->Connections[i][0];
		res += fabs(det(&(Mesh->Points[i]),&(Mesh->Points[j]),&(Mesh->Points[l])));
	}
	return (double)res/12.;
}

/*! 
 * @param i Matrixindex auf dem Gitter
 * @return quadratisches Überlappintegral mit Gradient von i
 */

point2D matrix_Aii_01(int i){
	int j,k,l;
	double detJ;
	double factor;
	point2D res;
	point2D* P1;
	point2D* P2;
	int len = glob_mesh.Sizes[i];
	res.x = 0;
	res.y = 0;
	point2D* P3 = &glob_mesh.Points[i];
	if (partition_mode && Partition[i]==0){
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[Partition[i]-1]; else factor = 1.;
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
	}
	vec_mult(&res,(double)1/6);
	return res;
}


/*! 
 * @param i,j Matrixindizes auf dem Gitter, i!=j
 * @return quadratisches Überlappintegral mit Gradient von j
 */

point2D matrix_Aij_01(int i,int j){
	point2D res;
	int len,k;
	double detJ;
	double factor;
	point2D* P1;
	point2D* P2 = &glob_mesh.Points[i];
	point2D* P3 = &glob_mesh.Points[j];
	int* list = overlap(i,j);
	if (list[1]<0){len = 1;}else{len = 2;}
	res.x = 0;
	res.y = 0;
	if (partition_mode && Partition[i]==0 && Partition[j]==0){
		for (k=0;k<len;k++){
			factor = Partition_attributes[Partition[list[k]]-1];
			P1 = &glob_mesh.Points[list[k]];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[max(Partition[i],Partition[j])-1]; else factor = 1.;
		for (k=0;k<len;k++){
			P1 = &glob_mesh.Points[list[k]];
			detJ = det(P1,P2,P3);
			res.x += ((P1->y)-(P2->y))*sign(detJ)*factor;
			res.y += -((P1->x)-(P2->x))*sign(detJ)*factor;
		}
	}
	free(list);
	vec_mult(&res,(double)1/6);
	return res;
}

/*! 
 * @param i,j Matrixindizes auf dem Gitter, i!=j
 * @return quadratisches Überlappintegral mit Gradient nach i und j
 */

matrix2D matrix_Aii_11(int i){
	int j,k,l;
	double detJ,factor;
	matrix2D res;
	point2D* P1;
	point2D* P2;
	int len = glob_mesh.Sizes[i];
	point2D* P3 = &glob_mesh.Points[i];
	res.xx = 0;
	res.yx = 0;
	res.xy = 0;
	res.yy = 0;
	if (partition_mode && Partition[i]==0){
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[Partition[i]-1]; else factor = 1.;
		for (k=0;k<len-1;k++){
			j = glob_mesh.Connections[i][k];
			l = glob_mesh.Connections[i][k+1];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
		if (glob_mesh.Is_boundary[i]<0){
			j = glob_mesh.Connections[i][len-1];
			l = glob_mesh.Connections[i][0];
			P1 = &glob_mesh.Points[j];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
	}
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Aij_11(int i,int j){
	matrix2D res;
	int len,k;
	double detJ,factor;
	point2D* P1;
	point2D* P2 = &glob_mesh.Points[i];
	point2D* P3 = &glob_mesh.Points[j];
	int* list = overlap(i,j);
	if (list[1]<0){len = 1;}else{len = 2;}
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	if (partition_mode && Partition[i]==0 && Partition[j]==0){
		for (k=0;k<len;k++){
			P1 = &glob_mesh.Points[list[k]];
			factor = Partition_attributes[Partition[list[k]]-1];
			detJ = fabs(det(P1,P2,P3));
			res.xx += -factor*((P1->y)-(P2->y))*((P1->y)-(P3->y))/detJ;
			res.xy += factor*((P1->x)-(P2->x))*((P1->y)-(P3->y))/detJ;
			res.yx += factor*((P1->x)-(P3->x))*((P1->y)-(P2->y))/detJ;
			res.yy += -factor*((P1->x)-(P2->x))*((P1->x)-(P3->x))/detJ;
		}
	}
	else{
		if (partition_mode) factor = Partition_attributes[max(Partition[i],Partition[j])-1]; else factor = 1.;
		for (k=0;k<len;k++){
			P1 = &glob_mesh.Points[list[k]];
			detJ = fabs(det(P1,P2,P3));
			res.xx += -factor*((P1->y)-(P2->y))*((P1->y)-(P3->y))/detJ;
			res.xy += factor*((P1->x)-(P2->x))*((P1->y)-(P3->y))/detJ;
			res.yx += factor*((P1->x)-(P3->x))*((P1->y)-(P2->y))/detJ;
			res.yy += -factor*((P1->x)-(P2->x))*((P1->x)-(P3->x))/detJ;
		}
	}
	mat_mult(&res,(double)1/2);
	free(list);
	return res;
}

matrix2D matrix_Biii_011(int i){
	matrix2D res = matrix_Aii_11(i);
	mat_mult(&res,(double)1/3);
	return res;
}

matrix2D matrix_Biij_011(int i,int j){
	matrix2D res = matrix_Aij_11(i,j);
	mat_mult(&res,(double)1/3);
	return res;
}

matrix2D matrix_Biji_011(int i,int j){
	matrix2D res = matrix_Aij_11(j,i);
	mat_mult(&res,(double)1/3);
	return res;
}

matrix2D matrix_Bijj_011(int i,int j){
	int k,l;
	double detJ,factor;
	point2D* P2;
	matrix2D res;
	int len = glob_mesh.Sizes[i];
	point2D* P1 = &glob_mesh.Points[i];
	point2D* P3 = &glob_mesh.Points[j];
	int* list = overlap(i,j);
	if (list[1]<0){len = 1;}else{len = 2;}
	res.xx = 0;
	res.yx = 0;
	res.xy = 0;
	res.yy = 0;
	if (partition_mode && Partition[i]==0){
		for (k=0;k<len;k++){
			l = list[k];
			factor = Partition_attributes[max(Partition[j],Partition[l])-1];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));			
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
		
	}
	else{
		if (partition_mode) factor = Partition_attributes[Partition[i]-1]; else factor = 1.;
		for (k=0;k<len;k++){
			l = list[k];
			P2 = &glob_mesh.Points[l];
			detJ = fabs(det(P1,P2,P3));
			res.xx += factor*((P1->y)-(P2->y))*((P1->y)-(P2->y))/detJ;
			res.xy += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yx += -factor*((P1->x)-(P2->x))*((P1->y)-(P2->y))/detJ;
			res.yy += factor*((P1->x)-(P2->x))*((P1->x)-(P2->x))/detJ;
		}
	}
	free(list);
	mat_mult(&res,(double)1/6);
	return res;
}

matrix2D matrix_Bijk_011(int i,int j,int k){
	matrix2D res;
	int len;
	double detJ,factor;
	point2D* P1;
	point2D* P2 = &glob_mesh.Points[j];
	point2D* P3 = &glob_mesh.Points[k];
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	if (partition_mode && Partition[j]==0 && Partition[k]==0){
		P1 = &glob_mesh.Points[i];
		factor = Partition_attributes[Partition[i]-1];
		detJ = fabs(det(P1,P2,P3));
		res.xx += -factor*((P1->y)-(P2->y))*((P1->y)-(P3->y))/detJ;
		res.xy += factor*((P1->x)-(P2->x))*((P1->y)-(P3->y))/detJ;
		res.yx += factor*((P1->x)-(P3->x))*((P1->y)-(P2->y))/detJ;
		res.yy += -factor*((P1->x)-(P2->x))*((P1->x)-(P3->x))/detJ;
		
	}
	else{
		if (partition_mode) factor = Partition_attributes[max(Partition[j],Partition[k])-1]; else factor = 1.;
		P1 = &glob_mesh.Points[i];
		detJ = fabs(det(P1,P2,P3));
		res.xx += -factor*((P1->y)-(P2->y))*((P1->y)-(P3->y))/detJ;
		res.xy += factor*((P1->x)-(P2->x))*((P1->y)-(P3->y))/detJ;
		res.yx += factor*((P1->x)-(P3->x))*((P1->y)-(P2->y))/detJ;
		res.yy += -factor*((P1->x)-(P2->x))*((P1->x)-(P3->x))/detJ;
	}
	mat_mult(&res,(double)1/6);
	return res;
}

point2D matrix_Biii_001(int i){
	point2D res = matrix_Aii_01(i);
	vec_mult(&res,(double)1/2);
	return res;
}

point2D matrix_Biij_001(int i,int j){
	point2D res = matrix_Aij_01(i,j);
	vec_mult(&res,(double)1/2);
	return res;
}

point2D matrix_Biji_001(int i,int j){
	int k;
	double detF;
	point2D grad;
	point2D res = init_point2D(0,0);
	int* list = overlap(i,j);
	
	k = list[0];
	grad = grad_2D(i,j,k);
	detF = fabs(det_tri(i,j,k));
	vec_add_mult(&res,&grad,detF);
	
	k = list[1];
	if (k>=0){
		grad = grad_2D(i,j,k);
		detF = fabs(det_tri(i,j,k));
		vec_add_mult(&res,&grad,detF);		
	}
	
	vec_mult(&res,(double)1/24);
	free(list);
	return res;
}

point2D matrix_Bijj_001(int i,int j){
	point2D res = matrix_Aij_01(i,j);
	vec_mult(&res,(double)1/4);
	return res;	
}

point2D matrix_Bijk_001(int i,int j,int k){
	double detF;
	point2D grad;
	point2D res = init_point2D(0,0);
	
	grad = grad_2D(k,i,j);
	detF = fabs(det_tri(i,j,k));
	vec_add_mult(&res,&grad,detF);
	
	vec_mult(&res,(double)1/24);
	return res;
}

double matrix_Biii_000(int i){
	return (double)3/5*matrix_Aii_00(i);
}

double matrix_Biij_000(int i,int j){
	return (double)2/5*matrix_Aij_00(i,j);
}

double matrix_Biji_000(int i,int j){
	return matrix_Biij_000(i,j);
}

double matrix_Bijj_000(int i,int j){
	return matrix_Biij_000(i,j);
}

double matrix_Bijk_000(int i,int j,int k){
	double detF = fabs(det_tri(i,j,k));
	return (double)1/120*detF;
}

matrix2D matrix_Ciiii_0011(int i){
	matrix2D res = matrix_Biii_011(i);
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Ciiij_0011(int i,int j){
	matrix2D res = matrix_Biij_011(i,j);
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Ciiji_0011(int i,int j){
	matrix2D res = matrix_Biji_011(i,j);
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Cijii_0011(int i,int j){
	matrix2D res = matrix_Bijj_011(j,i);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijjj_0011(int i,int j){
	matrix2D res = matrix_Bijj_011(i,j);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Ciijj_0011(int i,int j){
	matrix2D res = matrix_Bijj_011(i,j);
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Cijij_0011(int i,int j){
	matrix2D res = matrix_Biij_011(i,j);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijji_0011(int i,int j){
	matrix2D res = matrix_Biji_011(i,j);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Ciijk_0011(int i,int j,int k){
	matrix2D res = matrix_Bijk_011(i,j,k);
	mat_mult(&res,(double)1/2);
	return res;
}

matrix2D matrix_Cijik_0011(int i,int j,int k){
	matrix2D res = matrix_Bijk_011(j,i,k);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijkj_0011(int i,int j,int k){
	matrix2D res = matrix_Bijk_011(i,k,j);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijki_0011(int i,int j,int k){
	matrix2D res = matrix_Bijk_011(j,k,i);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijkk_0011(int i,int j,int k){
	matrix2D res = matrix_Bijj_011(i,k);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D matrix_Cijjk_0011(int i,int j,int k){
	matrix2D res = matrix_Bijk_011(i,j,k);
	mat_mult(&res,(double)1/4);
	return res;
}

matrix2D CB(int i){
	int j;
	matrix2D v;
	matrix2D res;
	res = matrix_Biii_011(i);
	for (j=0;j<glob_mesh.Sizes[i];j++){
		v = matrix_Cijjj_0011(glob_mesh.Connections[i][j],i);
		mat_add(&res,&v,-1.);
	}
	v = matrix_Ciiii_0011(i);
	mat_add(&res,&v,-1.);
	return res;
}

/*matrix2D CA(int i){
	int j,J;
	matrix2D v;
	matrix2D res;
	res = matrix_Aii_11(i);
	for (j=0;j<glob_mesh.Sizes[i];j++){
		J = glob_mesh.Connections[i][j];
		v = matrix_Cijjj_0011(J,i);
		mat_add(&res,&v,-1.);
		v = matrix_Cijii_0011(i,J);
		mat_add(&res,&v,-1.);
		v = matrix_Ciijj_0011(K,J,i);
		mat_add(&res,&v,-1.);
	}
	v = matrix_Ciiii_0011(i);
	mat_add(&res,&v,-1.);
	return res;
}*/

matrix2D matrix_bound_2_mixed_Aii_00(int* Cond,int i){
	int j1,j2;
	double a;
	matrix2D res;
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	if (is_boundary(i)>=0 && (Cond[i]==NEUMANN || Cond[i]==MIXED)){
		j1 = glob_mesh.Connections[i][0];
		j2 = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
		if (Cond[j1]==NEUMANN || Cond[j1]==MIXED){
			a = dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j1]));
			res.xx += a;
			res.yy += a;
		}
		if (Cond[j2]==NEUMANN || Cond[j2]==MIXED){
			a = dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j2]));
			res.xx += a;
			res.yy += a;
		}
	}
	mat_mult(&res,(double)1/3);
	return res;
}

matrix2D matrix_bound_2_mixed_Aij_00(int* Cond,int i,int j){
	int b;
	double a;
	matrix2D res;
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	b = (Cond[i]==NEUMANN || Cond[i]==MIXED) && (Cond[j]==NEUMANN || Cond[j]==MIXED);
	if (is_boundary(i)>=0 && is_boundary(j)>=0 && b){
		a = dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]))/6.;
		res.xx += a;
		res.yy += a;
	}
	
	return res;
}

double matrix_bound_0_Aii_00(int i){
	int j1,j2;
	double res = 0;
	if (is_boundary(i)>=0){
		j1 = glob_mesh.Connections[i][0];
		j2 = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
		res += dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j1]));
		res += dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j2]));
	}
	return res/3.;
}

double matrix_bound_0_Aij_00(int i,int j){
	double res = 0;
	if (is_boundary(i)>=0 && is_boundary(j)>=0){
		return dist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]))/6.;
	}
	else return 0;
}

double matrix_bound_0_Biii_000(int i){
	return (double)3/4*matrix_bound_0_Aii_00(i);
}

double matrix_bound_0_Biij_000(int i,int j){
	return (double)1/2*matrix_bound_0_Aij_00(i,j);
}

double matrix_bound_0_Biji_000(int i,int j){
	return matrix_bound_0_Biij_000(i,j);
}

double matrix_bound_0_Bijj_000(int i,int j){
	return matrix_bound_0_Biij_000(i,j);
}

point2D matrix_bound_1_Aii_00(int i){
	int j,k;
	int* list;
	double a,factor;
	point2D res;
	point2D nor;
	point2D* P;
	point2D* C = &glob_mesh.Points[i];
	res.x = 0;
	res.y = 0;
	if (partition_mode && Partition[i]==0){
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			P = &glob_mesh.Points[k];
			if (Partition[k]==0){
				list = overlap(i,k);
				if (Partition[list[0]]>0) factor = Partition_attributes[Partition[list[0]]-1];
				else{
					printf("ungünstige Partitionierung bei index %d\n",i);
					exit(0);
				};
				nor = get_normal(C,P,&glob_mesh.Points[list[0]]);
				a = factor*dist(C,P)/3.;
				res.x += a*nor.x;
				res.y += a*nor.y;
				if (list[1]>=0){
					if (Partition[list[1]]>0) factor = Partition_attributes[Partition[list[1]]-1];
					else{
						printf("ungünstige Partitionierung bei index %d\n",i);
						exit(0);
					}
					factor = Partition_attributes[Partition[list[1]]-1];
					nor = get_normal(C,P,&glob_mesh.Points[list[1]]);
					res.x += a*nor.x;
					res.y += a*nor.y;
				}
				free(list);
			}
		}
	}
	else if (is_boundary(i)>=0){
		P = &glob_mesh.Points[glob_mesh.Connections[i][0]];
		list = overlap(i,glob_mesh.Connections[i][0]);
		nor = get_normal(C,P,&glob_mesh.Points[list[0]]);
		vec_add_mult(&res,&nor,dist(C,P)/3.);
		free(list);
		P = &glob_mesh.Points[glob_mesh.Connections[i][glob_mesh.Sizes[i]-1]];
		list = overlap(i,glob_mesh.Connections[i][glob_mesh.Sizes[i]-1]);
		nor = get_normal(C,P,&glob_mesh.Points[list[0]]);
		vec_add_mult(&res,&nor,dist(C,P)/3.);
		free(list);
	}
	return res;
}


// computes \oint grad(phi_i)_a*phi_j*phi_k*ds_b
// i,j index on boundary node, k index of inner node

matrix2D matrix_bound_Biii_1_100(int i){
	int j,J,k,len;
	int* list;
	point2D grad;
	point2D norm;
	matrix2D* A;
	matrix2D res;
	len = glob_mesh.Sizes[i];
	
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;	

	if (is_boundary(i)>=0){
		j = glob_mesh.Connections[i][0];
		list = overlap(i,j);
		k = list[0];
		grad = grad_2D(i,j,k);
		norm = bound_line_2D(i,j,k);
		A = tensor_product(&grad,&norm);
		mat_add(&res,A,1.);
		free(list);
		free(A);
		
		j = glob_mesh.Connections[i][len-1];
		list = overlap(i,j);
		k = list[0];
		grad = grad_2D(i,j,k);
		norm = bound_line_2D(i,j,k);
		A = tensor_product(&grad,&norm);
		mat_add(&res,A,1.);
		free(list);
		free(A);
	}
	
	mat_mult(&res,(double)1/3);
	return res;
}


matrix2D matrix_bound_Biij_1_100(int i,int j){
	int k;
	int* list;
	point2D grad;
	point2D norm;
	matrix2D res;
	matrix2D* A;
	
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;	
	
	if (is_boundary(i)>=0 && is_boundary(j)>=0){
		list = overlap(i,j);
		k = list[0];
		grad = grad_2D(i,j,k);
		norm = bound_line_2D(i,j,k);
		A = tensor_product(&grad,&norm);
		mat_add(&res,A,1.);
		free(list);
		free(A);
	}
		
	mat_mult(&res,(double)1/6);
	return res;
}

matrix2D matrix_bound_Bijj_1_100(int i,int j){
	int k;
	int* list;
	point2D grad;
	point2D norm;
	matrix2D res;
	matrix2D* A;
	
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;	
	
	if (is_boundary(j)>=0){
		list = overlap(i,j);
		k = list[0];
		grad = grad_2D(i,j,k);
		if (is_boundary(i)>=0) norm = bound_line_2D(i,j,k);
		else{
			if (is_boundary(k)>=0) norm = bound_line_2D(j,k,i);
			else norm = init_point2D(0,0);
		} 
		A = tensor_product(&grad,&norm);
		mat_add(&res,A,1.);
		free(A);
		
		k = list[1];
		if (k>=0){
			grad = grad_2D(i,j,k);
			if (is_boundary(i)>=0) norm = bound_line_2D(i,j,k);
			else{
				if (is_boundary(k)>=0) norm = bound_line_2D(j,k,i);
				else norm = init_point2D(0,0);
			} 
			A = tensor_product(&grad,&norm);
			mat_add(&res,A,1.);
			free(A);					
		}
		free(list);
	}
	
	mat_mult(&res,(double)1/3);
	return res;
}

matrix2D matrix_bound_Bijk_1_100(int i,int j,int k){
	point2D grad;
	point2D norm;
	matrix2D res;
	matrix2D* A;
	
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;	
	
	if (is_boundary(j)>=0 && is_boundary(k)>=0){
		grad = grad_2D(i,j,k);
		norm = bound_line_2D(j,k,i);
		A = tensor_product(&grad,&norm);
		mat_add(&res,A,1.);		
		free(A);
	}
	
	mat_mult(&res,(double)1/6);
	return res;
}
	
matrix2D matrix_bound_Biji_1_100(int i,int j){
	return matrix_bound_Biij_1_100(i,j);
}

int* outer_triangle(int i,int j){
	int* Res;
	int* list = overlap(i,j);
	if (is_boundary(list[0])>=0){
		Res = zero_int_list(3);
		Res[0] = i;
		Res[1] = j;
		Res[2] = list[0];
		free(list);
		return Res;
	}
	if (list[1]>=0 && is_boundary(list[1])>=0){
		Res = zero_int_list(3);
		Res[0] = i;
		Res[1] = j;
		Res[2] = list[1];
		free(list);
		return Res;
	}
	free(list);
	return NULL;
}

// only if all three nodes are at boundary
void two_of_three_bound(int i,int j,int k,int* e,int* f,int* g){			// e is set to degree two node
	*e = (glob_mesh.Sizes[i]==2 ? i: (glob_mesh.Sizes[j]==2 ? j : k));
	if (*e==i){
		*f = j;
		*g = k;
	}
	if (*e==j){
		*f = i;
		*g = k;
	}
	if (*e==k){
		*f = j;
		*g = i;
	}
}

// only if all two nodes are at boundary
void one_of_three_inner(int i,int j,int k,int* e,int* f,int* g){			// e is set to inner node 
	if (is_boundary(i)<0){
		*e = i;
		*f = j;
		*g = k;
	}
	else{
		if (is_boundary(j)<0){
			*e = j;
			*f = i;
			*g = k;
		}
		else{
			*e = k;
			*f = i;
			*g = j;
		}
	}
}

sparse_matrix* set_matrix_bound_0_Aij_00_2D(int equ,int var,void (*insert)(sparse_matrix* A,double v,int i,int j,int d)){
	int I,i,j,k,bi,bj,bk,e,f,g;
	int n = glob_mesh.size;
	double ds,a;
	int* list;
	sparse_matrix* Res = sparse_zero(var_number_2D*n);
	for (I=0;I<elements.size;I++){
		i = elements.Elements[I].i;
		j = elements.Elements[I].j;
		k = elements.Elements[I].k;
		bi = is_boundary(i);
		bj = is_boundary(j);
		bk = is_boundary(k);
		
		if (bi>=0 && bj>=0 && bk>=0){
			two_of_three_bound(i,j,k,&e,&f,&g);						
																					// Aii																
			a = 0;															// ee
			a += dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[f]))/3.;				
			a += dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[g]))/3.;	
			(*insert)(Res,a,equ*n+e,var*n+e,n);			
																	
			a = dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[f]))/3.;		// ff
			(*insert)(Res,a,equ*n+f,var*n+f,n);		
			
			a = dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[g]))/3.;		// gg
			(*insert)(Res,a,equ*n+g,var*n+g,n);		
																					// Aij
			a = dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[f]))/6.;		// ef
			(*insert)(Res,a,equ*n+e,var*n+f,n);
			(*insert)(Res,a,equ*n+f,var*n+e,n);
					
			a = dist(&(glob_mesh.Points[e]),&(glob_mesh.Points[g]))/6.;		// eg
			(*insert)(Res,a,equ*n+e,var*n+g,n);
			(*insert)(Res,a,equ*n+g,var*n+e,n);	
																			// fg -> 0
			goto finished;
		}
		if ((bi>=0 && bj>=0) || (bj>=0 && bk>=0) || (bk>=0 && bi>=0)){
			one_of_three_inner(i,j,k,&e,&f,&g);
			list = overlap(f,g);
			if (list[1]>=0){
				free(list);
				goto finished;
			}
			else free(list);			
			
			ds = dist(&(glob_mesh.Points[f]),&(glob_mesh.Points[g]));	
																					// Aii				
			a = ds/3.;														// ff
			(*insert)(Res,a,equ*n+f,var*n+f,n);		
			
			a = ds/3.;														// gg
			(*insert)(Res,a,equ*n+g,var*n+g,n);					
																					// Aij
			a = ds/6.;														// fg
			(*insert)(Res,a,equ*n+f,var*n+g,n);
			(*insert)(Res,a,equ*n+g,var*n+f,n);
																			// ef -> 0
																			// eg -> 0			
		}		
		finished:;
	}
	return Res;
}

sparse_matrix* set_matrix_bound_1_Aij_00_2D(int equ,int var,void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d)){
	int I,i,j,k,bi,bj,bk,e,f,g;
	int n = glob_mesh.size;	
	point2D v;
	int* list;
	sparse_matrix* Res = sparse_zero(var_number_2D*n);
	for (I=0;I<elements.size;I++){
		i = elements.Elements[I].i;
		j = elements.Elements[I].j;
		k = elements.Elements[I].k;
		bi = is_boundary(i);
		bj = is_boundary(j);
		bk = is_boundary(k);
		
		if (bi>=0 && bj>=0 && bk>=0){
			two_of_three_bound(i,j,k,&e,&f,&g);						
																					// Aii																
														
			v = bound_line_2D(f,g,e);										// ee
			vec_mult(&v,(double)-1/3);
			(*insert)(Res,&v,equ*n+e,var*n+e,n);					
											
			v = bound_line_2D(f,e,g);										// ff				
			vec_mult(&v,(double)1/3);
			(*insert)(Res,&v,equ*n+f,var*n+f,n);		
			
			v = bound_line_2D(g,e,f);										// gg
			vec_mult(&v,(double)1/3);
			(*insert)(Res,&v,equ*n+g,var*n+g,n);		
																					// Aij
			v = bound_line_2D(f,e,g);										// ef
			vec_mult(&v,(double)1/6);
			(*insert)(Res,&v,equ*n+e,var*n+f,n);
			(*insert)(Res,&v,equ*n+f,var*n+e,n);
					
			v = bound_line_2D(g,e,f);										// eg
			vec_mult(&v,(double)1/6);
			(*insert)(Res,&v,equ*n+e,var*n+g,n);
			(*insert)(Res,&v,equ*n+g,var*n+e,n);	
																			// fg -> 0
			goto finished;
		}
		if ((bi>=0 && bj>=0) || (bj>=0 && bk>=0) || (bk>=0 && bi>=0)){
			one_of_three_inner(i,j,k,&e,&f,&g);
			list = overlap(f,g);
			if (list[1]>=0){
				free(list);
				goto finished;
			}
			else free(list);			
			
			v = bound_line_2D(f,g,e);	
			vec_mult(&v,(double)1/3);
																					// Aii			
			(*insert)(Res,&v,equ*n+f,var*n+f,n);							// ff
											
			(*insert)(Res,&v,equ*n+g,var*n+g,n);							// gg
																					// Aij
			vec_mult(&v,(double)1/2);										// fg
			(*insert)(Res,&v,equ*n+f,var*n+g,n);
			(*insert)(Res,&v,equ*n+g,var*n+f,n);
																			// ef -> 0
																			// eg -> 0			
		}		
		finished:;
	}
	return Res;
}

sparse_matrix* set_matrix_bound_2_Aij_10_2D(int equ,int var, void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int I,i,j,k,bi,bj,bk,e,f,g;
	int n = glob_mesh.size;
	int* list;
	point2D grad;
	point2D ds;
	matrix2D* A;
	matrix2D* B;
	sparse_matrix* Res = sparse_zero(var_number_2D*n);
	for (I=0;I<elements.size;I++){
		i = elements.Elements[I].i;
		j = elements.Elements[I].j;
		k = elements.Elements[I].k;
		bi = is_boundary(i);
		bj = is_boundary(j);
		bk = is_boundary(k);
		
		if (bi>=0 && bj>=0 && bk>=0){
			two_of_three_bound(i,j,k,&e,&f,&g);
			
			grad = grad_2D(e,f,g);					// ee
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+e,var*n+e,n);
			free(A);
			
			grad = grad_2D(f,g,e);					// ff
			ds = bound_line_2D(f,e,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+f,var*n+f,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// gg
			ds = bound_line_2D(g,e,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+g,var*n+g,n);
			free(A);
			
			grad = grad_2D(e,f,g);					// ef
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+e,var*n+f,n);
			free(A);
			
			grad = grad_2D(f,e,g);					// fe
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+f,var*n+e,n);
			free(A);
			
			grad = grad_2D(e,f,g);					// eg
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+e,var*n+g,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// ge
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+g,var*n+e,n);
			free(A);
			
			grad = grad_2D(f,e,g);					// fg
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+f,var*n+g,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// gf
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+g,var*n+f,n);
			free(A);
			
			goto finished;
		}
		
		if ((bi>=0 && bj>=0) || (bj>=0 && bk>=0) || (bk>=0 && bi>=0)){
			one_of_three_inner(i,j,k,&e,&f,&g);
			list = overlap(f,g);
			if (list[1]>=0){
				free(list);
				goto finished;
			}
			else free(list);			
			ds = bound_line_2D(f,g,e);
			
			grad = grad_2D(f,g,e);					// ff
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+f,var*n+f,n);
			free(A);
			
			A = tensor_product(&grad,&ds);			// fg
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+f,var*n+g,n);
			free(A);
						
			grad = grad_2D(g,e,f);					// gg
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+g,var*n+g,n);
			free(A);
										
			A = tensor_product(&grad,&ds);			// gf
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+g,var*n+f,n);
			free(A);
			
			grad = grad_2D(e,f,g);					// ef
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+e,var*n+f,n);
			free(A);
			
			A = tensor_product(&grad,&ds);			// eg
			mat_mult(A,(double)1/2);
			(*insert)(Res,A,equ*n+e,var*n+g,n);
			free(A);
			
			goto finished;
		}
		finished:;
	}
	
	return Res;
}

sparse_matrix3D* set_matrix_bound_2_Bijk_100_2D(int equ,int var1,int var2,void (*insert)(sparse_matrix3D* A,matrix2D* v,int i,int j,int k,int d)){
	int I,i,j,k,bi,bj,bk,e,f,g;
	int n = glob_mesh.size;
	int* list;
	point2D grad;
	point2D ds;
	matrix2D* A;
	matrix2D* B;
	sparse_matrix3D* Res = sparse_zero3D(var_number_2D*n);
	for (I=0;I<elements.size;I++){
		i = elements.Elements[I].i;
		j = elements.Elements[I].j;
		k = elements.Elements[I].k;
		bi = is_boundary(i);
		bj = is_boundary(j);
		bk = is_boundary(k);
		
		if (bi>=0 && bj>=0 && bk>=0){
			two_of_three_bound(i,j,k,&e,&f,&g);
																//Biii
			grad = grad_2D(e,f,g);					// eee
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+e,var1*n+e,var2*n+e,n);
			free(A);
			
			grad = grad_2D(f,g,e);					// fff
			ds = bound_line_2D(f,e,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+f,var1*n+f,var2*n+f,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// ggg
			ds = bound_line_2D(g,e,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+g,var1*n+g,var2*n+g,n);
			free(A);
			
																// Biij
																// Biji
			grad = grad_2D(e,f,g);					// eef
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+e,var1*n+e,var2*n+f,n);
			(*insert)(Res,A,equ*n+e,var1*n+f,var2*n+e,n);
			free(A);
			
			grad = grad_2D(f,e,g);					// ffe
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+f,var1*n+f,var2*n+e,n);
			(*insert)(Res,A,equ*n+f,var1*n+e,var2*n+f,n);
			free(A);
			
			grad = grad_2D(e,f,g);					// eeg
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+e,var1*n+e,var2*n+g,n);
			(*insert)(Res,A,equ*n+e,var1*n+g,var2*n+e,n);
			free(A);
											
			grad = grad_2D(g,e,f);					// gge
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+g,var1*n+g,var2*n+e,n);
			(*insert)(Res,A,equ*n+g,var1*n+e,var2*n+g,n);
			free(A);		
													// ggf -> 0
													// ffg -> 0
													
																// Bijj
			grad = grad_2D(f,e,g);					// fee
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+f,var1*n+e,var2*n+e,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// gee
			ds = bound_line_2D(f,g,e);
			vec_mult(&ds,-1.);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+g,var1*n+e,var2*n+e,n);
			free(A);
																
			grad = grad_2D(e,f,g);					// eff
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+e,var1*n+f,var2*n+f,n);
			free(A);
			
			grad = grad_2D(g,f,e);					// gff
			ds = bound_line_2D(e,f,g);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+g,var1*n+f,var2*n+f,n);
			free(A);
			
			grad = grad_2D(e,g,f);					// egg
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+e,var1*n+g,var2*n+g,n);
			free(A);
			
			grad = grad_2D(f,g,e);					// fgg
			ds = bound_line_2D(e,g,f);
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+f,var1*n+g,var2*n+g,n);
			free(A);
			
																// Bijk
			grad = grad_2D(g,e,f);					// gef
			ds = bound_line_2D(e,f,g);				
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+g,var1*n+e,var2*n+f,n);
			(*insert)(Res,A,equ*n+g,var1*n+f,var2*n+e,n);
			free(A);
			
			grad = grad_2D(f,e,g);					// feg
			ds = bound_line_2D(e,g,f);				
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+f,var1*n+e,var2*n+g,n);
			(*insert)(Res,A,equ*n+f,var1*n+g,var2*n+e,n);
			free(A);
													// efg	-> 0													
			goto finished;
		}
		if ((bi>=0 && bj>=0) || (bj>=0 && bk>=0) || (bk>=0 && bi>=0)){
			one_of_three_inner(i,j,k,&e,&f,&g);
			list = overlap(f,g);
			if (list[1]>=0){
				free(list);
				goto finished;
			}
			else free(list);		
			ds = bound_line_2D(f,g,e);
																//Biii
													// eee -> 0
			
			grad = grad_2D(f,g,e);					// fff			
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+f,var1*n+f,var2*n+f,n);
			free(A);
			
			grad = grad_2D(g,e,f);					// ggg			
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+g,var1*n+g,var2*n+g,n);
			free(A);			
																// Biij
																// Biji
													// eef -> 0						
													// ffe -> 0						
													// eeg -> 0											
													// gge -> 0
													
			grad = grad_2D(g,f,e);					// ggf			
			A = tensor_product(&grad,&ds);			
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+g,var1*n+g,var2*n+f,n);
			(*insert)(Res,A,equ*n+g,var1*n+f,var2*n+g,n);
			free(A);		
			
			grad = grad_2D(f,g,e);					// ffg		
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+f,var1*n+f,var2*n+g,n);
			(*insert)(Res,A,equ*n+f,var1*n+g,var2*n+f,n);
			free(A);	
																// Bijj															
													// fee -> 0
													// gee -> 0
													
			grad = grad_2D(e,f,g);					// eff
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+e,var1*n+f,var2*n+f,n);
			free(A);	
			
			grad = grad_2D(e,f,g);					// egg
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+e,var1*n+g,var2*n+g,n);
			free(A);	
			
			grad = grad_2D(g,f,e);					// gff
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+g,var1*n+f,var2*n+f,n);
			free(A);
			
			grad = grad_2D(f,g,e);					// fgg		
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/3);
			(*insert)(Res,A,equ*n+f,var1*n+g,var2*n+g,n);
			free(A);			
																// Bijk
													// fge -> 0
													// gef -> 0
													
			grad = grad_2D(e,f,g);					// efg
			A = tensor_product(&grad,&ds);
			mat_mult(A,(double)1/6);
			(*insert)(Res,A,equ*n+e,var1*n+f,var2*n+g,n);
			(*insert)(Res,A,equ*n+e,var1*n+g,var2*n+f,n);
			free(A);																																																			
		}	
		finished:;
	}
	return Res;
}

/*matrix2D matrix_bound_2_Aii_01(int i){
	return matrix_bound_2_Aii_10(i);
}

matrix2D matrix_bound_2_Aij_01(int i,int j){
	int k;
	int* list;
	matrix2D* A;
	matrix2D res;
	point2D ds;
	point2D grad;
	
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;	
	
	list = overlap(i,j);
	k= list[0];
	if (list[1]<0){
		grad = grad_2D(j,i,k);
		ds = bound_line_2D(i,j,k);
		A = tensor_product(&grad,&ds);
		mat_add(&res,A,1.);
		free(A);	
	}
	else{		
		if (is_boundary(k)>=0){
			grad = grad_2D(j,i,k);
			ds = bound_line_2D(i,k,j);
			A = tensor_product(&grad,&ds);
			mat_add(&res,A,1.);
			free(A);				
		}
		k = list[1];
		if (is_boundary(k)>=0){
			grad = grad_2D(j,i,k);
			ds = bound_line_2D(i,k,j);
			A = tensor_product(&grad,&ds);
			mat_add(&res,A,1.);
			free(A);				
		}
	}
	free(list);
	
	mat_mult(&res,(double)1/2);
	return res;	
}

sparse_matrix3D* set_matrix_bound_2_Bijk_100_2D(int equ,int var1,int var2,
  void (*insert)(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d)){
	int i,j,J,k,m,index_i,index_j,index_k;
	int* list;
	int d = glob_mesh.size;
	sparse_matrix3D* Res = sparse_zero3D(d*var_number_2D);				
	
	for (i=0;i<d;i++){
		if (is_boundary(i)>=0 || near_boundary(i)){								
			matrix2D v;
			index_i = d*equ+i;												// Biii
			index_j = d*var1+i;
			index_k = d*var2+i;
			v = matrix_bound_Biii_1_100(i);						
			(*insert)(Res,&v,index_i,index_j,index_k,d);
		
			for (J=0;J<glob_mesh.Sizes[i];J++){
				j = glob_mesh.Connections[i][J];
				if (is_boundary(j)>=0){
					index_j = d*var1+i;										// Biij
					index_k = d*var2+j;
					v = matrix_bound_Biij_1_100(i,j);						
					(*insert)(Res,&v,index_i,index_j,index_k,d);			
		
					index_j = d*var1+j;										// Biji
					index_k = d*var2+i;
					v = matrix_bound_Biji_1_100(i,j);						
					(*insert)(Res,&v,index_i,index_j,index_k,d);
			
					index_j = d*var1+j;										// Bijj
					index_k = d*var2+j;
					v = matrix_bound_Bijj_1_100(i,j);						
					(*insert)(Res,&v,index_i,index_j,index_k,d);
					
					list = overlap(i,j);
					for (k=0;k<2;k++) if (list[k]>=0){
						index_j = d*var1+j;									// Bijk
						index_k = d*var2+list[k];
						v = matrix_bound_Bijk_1_100(i,j,list[k]);						
						(*insert)(Res,&v,index_i,index_j,index_k,d);
					}								
					free(list);
				}
			}
		}	
	}
	return Res;
}

sparse_matrix* set_matrix_bound_2_Aij_10_2D(int equ,int var,
 void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int i,j,J,index_i,index_j,b1,b2;
	matrix2D v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	
	for (i=0;i<d;i++){
		if (is_boundary(i)>=0 || near_boundary(i)){	
			index_i = d*equ+i;
			index_j = d*var+i;
			v = matrix_bound_2_Aii_10(i);
			(*insert)(Res,&v,index_i,index_j,d);
		
			for (J=0;J<glob_mesh.Sizes[i];J++){
				j = glob_mesh.Connections[i][J];
				if (is_boundary(j)>=0){	
					index_j = d*var+j;
					v = matrix_bound_2_Aij_10(i,j);
					(*insert)(Res,&v,index_i,index_j,d);
				}
			}
		}
	}
	return Res;
}

sparse_matrix* set_matrix_bound_2_mixed_Aij_10_2D(int* Cond,int equ,int var,  // only nonzero if Dirichlet conditions present
 void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int i,j,J,index_i,index_j,b1,b2;
	matrix2D v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	
	for (i=0;i<d;i++){
		if (is_boundary(i)>=0 || near_boundary(i)){	
			index_i = d*equ+i;
			index_j = d*var+i;
			v = matrix_bound_2_mixed_Aii_10(Cond,i);
			(*insert)(Res,&v,index_i,index_j,d);
		
			for (J=0;J<glob_mesh.Sizes[i];J++){
				j = glob_mesh.Connections[i][J];
				if (is_boundary(j)>=0){	
					index_j = d*var+j;
					v = matrix_bound_2_mixed_Aij_10(Cond,i,j);
					(*insert)(Res,&v,index_i,index_j,d);
				}
			}
		}
	}
	return Res;
}

sparse_matrix* set_matrix_bound_2_mixed_Aij_00_2D(int* Cond, int equ,int var,
 void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int i,j,k,index_i,index_j;
	matrix2D v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_bound_2_mixed_Aii_00(Cond,i);
		(*insert)(Res,&v,index_i,index_j,d);
		for (k=0;k<glob_mesh.Sizes[i];k++){
			j = glob_mesh.Connections[i][k];
			if (is_boundary(j)>=0){
				index_j = d*var+j;
				v = matrix_bound_2_mixed_Aij_00(Cond,i,j);
				(*insert)(Res,&v,index_i,index_j,d);
			}
		}
	}
	return Res;
}

sparse_matrix* set_matrix_bound_2_Aij_01_2D(int equ,int var,
 void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int i,j,J,index_i,index_j;
	matrix2D v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++) if (is_boundary(i)>=0){	
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_bound_2_Aii_01(i);
		(*insert)(Res,&v,index_i,index_j,d);
		
		for (J=0;J<glob_mesh.Sizes[i];J++){
			j = glob_mesh.Connections[i][J];
			if (near_boundary(j)){	
				index_j = d*var+j;
				v = matrix_bound_2_Aij_01(i,j);
				(*insert)(Res,&v,index_i,index_j,d);
			}
		}
	}
	return Res;
}

sparse_matrix3D* set_matrix_bound_0_Bijk_000_2D(int equ,int var1,int var2,
  void (*insert)(sparse_matrix3D* B,double v,int i,int j,int k,int d)){
	int i,j,J,index_i,index_j,index_k;
	double v;
	int* list;
	int d = glob_mesh.size;
	sparse_matrix3D* Res = sparse_zero3D(d*var_number_2D);				
	
	for (i=0;i<d;i++) if (is_boundary(i)>=0){						
		index_i = d*equ+i;												// Biii
		index_j = d*var1+i;
		index_k = d*var2+i;
		v = matrix_bound_0_Biii_000(i);				
		(*insert)(Res,v,index_i,index_j,index_k,d);

		for (J=0;J<glob_mesh.Sizes[i];J++){
			j = glob_mesh.Connections[i][J];
			if (is_boundary(j)>=0){
				index_j = d*var1+i;										// Biij
				index_k = d*var2+j;
				v = matrix_bound_0_Biij_000(i,j);						
				(*insert)(Res,v,index_i,index_j,index_k,d);			
		
				index_j = d*var1+j;										// Biji
				index_k = d*var2+i;
				v = matrix_bound_0_Biji_000(i,j);						
				(*insert)(Res,v,index_i,index_j,index_k,d);
			
				index_j = d*var1+j;										// Bijj
				index_k = d*var2+j;
				v = matrix_bound_0_Bijj_000(i,j);						
				(*insert)(Res,v,index_i,index_j,index_k,d);																			
			}
		}
	}
	return Res;  
}

point2D matrix_bound_1_Aij_00(int i,int j){
	double a,factor;
	point2D res;
	point2D nor;
	point2D* C = &glob_mesh.Points[i];
	point2D* P = &glob_mesh.Points[j];
	int* list = overlap(i,j);
	res.x = 0;
	res.y = 0;
	if (partition_mode && Partition[i]==0 && Partition[j]==0){
		if (Partition[list[0]]>0) factor = Partition_attributes[Partition[list[0]]-1];
		else{
			printf("ungünstige Partitionierung bei index %d\n",i);
			exit(0);
		};
		nor = get_normal(C,P,&glob_mesh.Points[list[0]]);
		a = factor*dist(C,P)/6.;
		res.x += a*nor.x;
		res.y += a*nor.y;
		if (Partition[list[1]]>0) factor = Partition_attributes[Partition[list[1]]-1];
		else{
			printf("ungünstige Partitionierung bei index %d\n",i);
			exit(0);
		};
		nor = get_normal(C,P,&glob_mesh.Points[list[1]]);
		a = factor*dist(C,P)/6.;
		res.x += a*nor.x;
		res.y += a*nor.y;
	}
	else if (is_boundary(i)>=0 && is_boundary(j)>=0){
		nor = get_normal(C,P,&glob_mesh.Points[list[0]]);
		a = dist(C,P)/6.;
		res.x += a*nor.x;
		res.y += a*nor.y;
	}
	return res;
}

double matrix_bound_0_Aii_01(int i){
	int j,k;
	double a,res,factor;
	int* list = NULL;
	point2D nor;
	point2D* P1;
	point2D* P2;
	point2D* P3 = &glob_mesh.Points[i];
	res = 0;
	if (is_boundary(i)>=0){
		P2 = &glob_mesh.Points[glob_mesh.Connections[i][0]];
		P1 = &glob_mesh.Points[glob_mesh.Connections[i][1]];
		nor = get_normal(P3,P2,P1);
		a = dist(P2,P3)/det(P1,P2,P3);
		res += ((P1->y-P2->y)*nor.x-(P1->x-P2->x)*nor.y)*a/2;
		if (glob_mesh.Sizes[i]<2){printf("Fehler bei Randmatrix: %d\n",i);}
		P2 = &glob_mesh.Points[glob_mesh.Connections[i][glob_mesh.Sizes[i]-1]];
		P1 = &glob_mesh.Points[glob_mesh.Connections[i][glob_mesh.Sizes[i]-2]];
		nor = get_normal(P3,P2,P1);
		a = dist(P2,P3)/det(P1,P2,P3);
		res += ((P1->y-P2->y)*nor.x-(P1->x-P2->x)*nor.y)*a/2.;
	}
	else if (partition_mode && Partition[i]==0){
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (Partition[k]==0){
				list = overlap(i,k);
				P1 = &glob_mesh.Points[list[0]];
				P2 = &glob_mesh.Points[k];
				nor = get_normal(P3,P2,P1);
				if (Partition[list[0]]>0) factor = Partition_attributes[Partition[list[0]]-1];
				else{
					printf("ungünstige Partitionierung bei index %d\n",i);
					exit(0);
				}
				a = factor*dist(P2,P3)/det(P1,P2,P3);
				res += ((P1->y-P2->y)*nor.x-(P1->x-P2->x)*nor.y)*a/2.;
				if (list[1]>=0){
					P1 = &glob_mesh.Points[list[1]];
					vec_mult(&nor,-1.);
					if (Partition[list[1]]>0) factor = Partition_attributes[Partition[list[1]]-1];
					else{
						printf("ungünstige Partitionierung bei index %d\n",i);
						exit(0);
					}
					a = factor*dist(P2,P3)/det(P1,P2,P3);
					res += ((P1->y-P2->y)*nor.x-(P1->x-P2->x)*nor.y)*a/2.;
				}
				free(list);
			}
		}
	}
	return res;
}

double matrix_bound_0_Aij_01(int i,int j){  // nicht vertauschbar Aij_10
	double a,res,factor;
	point2D nor;
	point2D* P1 = NULL;
	point2D* P2 = NULL;
	point2D* P3 = &glob_mesh.Points[i];
	int* list = overlap(i,j);
	res = 0;
	if (is_boundary(i)>=0){
		if (is_boundary(j)>=0){
			P1 = &glob_mesh.Points[list[0]];
			P2 = &glob_mesh.Points[j];
			nor = get_normal(P3,P2,P1);
			a = dist(P2,P3)/det(P1,P2,P3);
			res += ((P3->y-P1->y)*nor.x-(P3->x-P1->x)*nor.y)*a/2.;
		}
		else{
			P1 = &glob_mesh.Points[j];
			if (is_boundary(list[0])>=0){
				P2 = &glob_mesh.Points[list[0]];
				nor = get_normal(P3,P2,P1);
				a = dist(P2,P3)/det(P1,P2,P3);
				res += ((P2->y-P3->y)*nor.x+(P3->x-P2->x)*nor.y)*a/2.;
			}
			if (list[1]>=0 && is_boundary(list[1])>=0){
				P2 = &glob_mesh.Points[list[1]];
				nor = get_normal(P3,P2,P1);
				a = dist(P2,P3)/det(P1,P2,P3);
				res += ((P2->y-P3->y)*nor.x+(P3->x-P2->x)*nor.y)*a/2.;
			}
		}
	}
	else if (Partition[i]==0){
		if (Partition[j]==0){
			P1 = &glob_mesh.Points[list[0]];
			P2 = &glob_mesh.Points[j];
			nor = get_normal(P3,P2,P1);
			if (Partition[list[0]]>0) factor = Partition_attributes[Partition[list[0]]-1];
			else{
				printf("ungünstige Partitionierung bei index %d\n",i);
				exit(0);
			}
			a = factor*dist(P2,P3)/det(P1,P2,P3);
			res += ((P3->y-P1->y)*nor.x-(P3->x-P1->x)*nor.y)*a/2.;
			P1 = &glob_mesh.Points[list[1]];
			vec_mult(&nor,-1.);
			if (Partition[list[1]]>0) factor = Partition_attributes[Partition[list[1]]-1];
			else{
				printf("ungünstige Partitionierung bei index %d\n",i);
				exit(0);
			}
			a = factor*dist(P2,P3)/det(P1,P2,P3);
			res += ((P3->y-P1->y)*nor.x-(P3->x-P1->x)*nor.y)*a/2.;
		}
		else{
			P1 = &glob_mesh.Points[j];
			factor = Partition_attributes[Partition[j]-1];
			if (Partition[list[0]]==0){
				P2 = &glob_mesh.Points[list[0]];
				nor = get_normal(P3,P2,P1);
				a = factor*dist(P2,P3)/det(P1,P2,P3);
				res += ((P2->y-P3->y)*nor.x+(P3->x-P2->x)*nor.y)*a/2.;
			}
			if (list[1]>=0 && Partition[list[1]]==0){
				P2 = &glob_mesh.Points[list[1]];
				nor = get_normal(P3,P2,P1);
				a = factor*dist(P2,P3)/det(P1,P2,P3);
				res += ((P2->y-P3->y)*nor.x+(P3->x-P2->x)*nor.y)*a/2.;
			}
		}
	}
	free(list);
	return res;
}*/

/*matrix2D matrix_bound_2_Aii_01(int i){
	int j,k,l1,l2;
	double a,factor;
	matrix2D res;
	int* list = NULL;
	point2D nor;
	point2D* P1;
	point2D* P2;
	point2D* P3 = &glob_mesh.Points[i];
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	if (is_boundary(i)>=0){
		P2 = &glob_mesh.Points[glob_mesh.Connections[i][0]];
		P1 = &glob_mesh.Points[glob_mesh.Connections[i][1]];
		nor = get_normal(P3,P2,P1);
		a = dist(P2,P3)/det(P1,P2,P3);
		res.xx += (P1->y-P2->y)*nor.x*a;
		res.xy += (P1->y-P2->y)*nor.y*a;
		res.yx += -(P1->x-P2->x)*nor.x*a;
		res.yy += -(P1->x-P2->x)*nor.y*a;
		if (glob_mesh.Sizes[i]<2){printf("Fehler bei Randmatrix: %d\n",i);}
		P2 = &glob_mesh.Points[glob_mesh.Connections[i][glob_mesh.Sizes[i]-1]];
		P1 = &glob_mesh.Points[glob_mesh.Connections[i][glob_mesh.Sizes[i]-2]];
		nor = get_normal(P3,P2,P1);
		a = dist(P2,P3)/det(P1,P2,P3);
		res.xx += (P1->y-P2->y)*nor.x*a;
		res.xy += (P1->y-P2->y)*nor.y*a;
		res.yx += -(P1->x-P2->x)*nor.x*a;
		res.yy += -(P1->x-P2->x)*nor.y*a;
	}
	else if (partition_mode && Partition[i]==0){
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (Partition[k]==0){
				list = overlap(i,k);
				l1 = -1;
				l2 = -1;
				if (Partition[list[0]]==Partition[i]){
					l1 = list[0];
					l2 = list[1];
				}
				if (Partition[list[1]]==Partition[i]){
					l1 = list[1];
					l2 = list[0];
				}
				if (l1>=0){
					P1 = &glob_mesh.Points[l1];
					P2 = &glob_mesh.Points[k];
					nor = get_normal(P3,P2,P1);
					factor = Partition_attributes[Partition[l1]-1];
					a = factor*dist(P2,P3)/det(P1,P2,P3);
					res.xx += (P1->y-P2->y)*nor.x*a;
					res.xy += (P1->y-P2->y)*nor.y*a;
					res.yx += -(P1->x-P2->x)*nor.x*a;
					res.yy += -(P1->x-P2->x)*nor.y*a;
					P1 = &glob_mesh.Points[l2];
					vec_mult(&nor,-1.);
					factor = Partition_attributes[Partition[l2]-1];
					a = factor*dist(P2,P3)/det(P1,P2,P3);
					res.xx += (P1->y-P2->y)*nor.x*a;
					res.xy += (P1->y-P2->y)*nor.y*a;
					res.yx += -(P1->x-P2->x)*nor.x*a;
					res.yy += -(P1->x-P2->x)*nor.y*a;
				}
				free(list);
			}
		}
	}
	mat_mult(&res,(double)1./2.);
	return res;
}*/

/*matrix2D matrix_bound_2_Aij_01(int i,int j){  // nicht vertauschbar Aij_10
	double a,factor;
	matrix2D res;
	point2D nor;
	point2D* P1 = NULL;
	point2D* P2 = NULL;
	point2D* P3 = &glob_mesh.Points[i];
	int* list = overlap(i,j);
	res.xx = 0;
	res.xy = 0;
	res.yx = 0;
	res.yy = 0;
	if (is_boundary(i)>=0){
		if (is_boundary(j)>=0){
			P1 = &glob_mesh.Points[list[0]];
			P2 = &glob_mesh.Points[j];
			nor = get_normal(P3,P2,P1);
			a = dist(P2,P3)/det(P1,P2,P3);
			res.xx += (P1->y-P2->y)*nor.x*a;
			res.xy += (P1->y-P2->y)*nor.y*a;
			res.yx += -(P1->x-P2->x)*nor.x*a;
			res.yy += -(P1->x-P2->x)*nor.y*a;
		}
		else{
			P1 = &glob_mesh.Points[j];
			if (is_boundary(list[0])>=0){
				P2 = &glob_mesh.Points[list[0]];
				nor = get_normal(P3,P2,P1);
				a = dist(P2,P3)/det(P1,P2,P3);
				res.xx += (P1->y-P2->y)*nor.x*a;
				res.xy += (P1->y-P2->y)*nor.y*a;
				res.yx += -(P1->x-P2->x)*nor.x*a;
				res.yy += -(P1->x-P2->x)*nor.y*a;
			}
			if (list[1]>=0 && is_boundary(list[1])>=0){
				P2 = &glob_mesh.Points[list[1]];
				nor = get_normal(P3,P2,P1);
				a = dist(P2,P3)/det(P1,P2,P3);
				res.xx += (P1->y-P2->y)*nor.x*a;
				res.xy += (P1->y-P2->y)*nor.y*a;
				res.yx += -(P1->x-P2->x)*nor.x*a;
				res.yy += -(P1->x-P2->x)*nor.y*a;
			}
		}
	}
	else if (Partition[i]==0){
		if (Partition[j]==0){
			P1 = &glob_mesh.Points[list[0]];
			P2 = &glob_mesh.Points[j];
			nor = get_normal(P3,P2,P1);
			if (Partition[list[0]]>0) factor = Partition_attributes[Partition[list[0]]-1];
			else{
				printf("ungünstige Partitionierung bei index %d\n",i);
				exit(0);
			}
			a = factor*dist(P2,P3)/det(P1,P2,P3);
			res.xx += (P1->y-P2->y)*nor.x*a;
			res.xy += (P1->y-P2->y)*nor.y*a;
			res.yx += -(P1->x-P2->x)*nor.x*a;
			res.yy += -(P1->x-P2->x)*nor.y*a;
			P1 = &glob_mesh.Points[list[1]];
			vec_mult(&nor,-1.);
			if (Partition[list[1]]>0) factor = Partition_attributes[Partition[list[1]]-1];
			else{
				printf("ungünstige Partitionierung bei index %d\n",i);
				exit(0);
			}
			a = factor*dist(P2,P3)/det(P1,P2,P3);
			res.xx += (P1->y-P2->y)*nor.x*a;
			res.xy += (P1->y-P2->y)*nor.y*a;
			res.yx += -(P1->x-P2->x)*nor.x*a;
			res.yy += -(P1->x-P2->x)*nor.y*a;
		}
		else{
			P1 = &glob_mesh.Points[j];
			factor = Partition_attributes[Partition[j]-1];
			if (Partition[list[0]]==0){
				P2 = &glob_mesh.Points[list[0]];
				nor = get_normal(P3,P2,P1);
				a = factor*dist(P2,P3)/det(P1,P2,P3);
				res.xx += (P1->y-P2->y)*nor.x*a;
				res.xy += (P1->y-P2->y)*nor.y*a;
				res.yx += -(P1->x-P2->x)*nor.x*a;
				res.yy += -(P1->x-P2->x)*nor.y*a;
			}
			if (list[1]>=0 && Partition[list[1]]==0){
				P2 = &glob_mesh.Points[list[1]];
				nor = get_normal(P3,P2,P1);
				a = factor*dist(P2,P3)/det(P1,P2,P3);
				res.xx += (P1->y-P2->y)*nor.x*a;
				res.xy += (P1->y-P2->y)*nor.y*a;
				res.yx += -(P1->x-P2->x)*nor.x*a;
				res.yy += -(P1->x-P2->x)*nor.y*a;
			}
		}
	}
	free(list);
	mat_mult(&res,(double)1./2.);
	return res;
}*/


/*point2D matrix_bound_1_Biii_000(int i){					// Achtung: B-Elemente noch nicht für Partition geeignet !
	point2D res = matrix_bound_1_Aii_00(i);
	vec_mult(&res,(double)3/4);
	return res;
}


point2D matrix_bound_1_Biij_000(int i,int j){
	point2D res = matrix_bound_1_Aij_00(i,j);
	vec_mult(&res,(double)1/2);
	return res;
}


double matrix_bound_0_Biii_001(int i){
	double res = matrix_bound_0_Aii_01(i);
	return (double)2/3*res;
}

matrix2D matrix_bound_2_Biii_001(int i){
	matrix2D res = matrix_bound_2_Aii_01(i);
	mat_mult(&res,(double)2/3);
	return res;
}


double matrix_bound_0_Bijj_001(int i,int j){
	double res = matrix_bound_0_Aij_01(i,j);
	return (double)1/3*res;
}

matrix2D matrix_bound_2_Bijj_001(int i,int j){
	matrix2D res = matrix_bound_2_Aij_01(i,j);
	mat_mult(&res,(double)1/3);
	return res;
}


double matrix_bound_0_Biij_001(int i,int j){
	double res = matrix_bound_0_Aij_01(i,j);
	return (double)2/3*res;
}

matrix2D matrix_bound_2_Biij_001(int i,int j){
	matrix2D res = matrix_bound_2_Aij_01(i,j);
	mat_mult(&res,(double)2/3);
	return res;
}

double matrix_bound_0_Bijk_001(int i,int j){
	double res;
	point2D nor;
	int* list = overlap(i,j);
	if (list[1]>0){printf("Gitterfehler bei %d:%d\n",i,j);}
	point2D* P1 = &glob_mesh.Points[i];
	point2D* P2 = &glob_mesh.Points[j];
	point2D* P3 = &glob_mesh.Points[list[0]];
	nor = get_normal(P1,P2,P3);
	double a = dist(P1,P2)/det(P1,P2,P3);
	res = (-((P2->y)-(P1->y))*nor.x+((P2->x)-(P1->x))*nor.y)*a/6;
	free(list);
	return res;
}

matrix2D matrix_bound_2_Bijk_001(int i,int j){
	point2D nor;
	matrix2D res;
	int* list = overlap(i,j);
	if (list[1]>0){printf("Gitterfehler bei %d:%d\n",i,j);}
	point2D* P1 = &glob_mesh.Points[i];
	point2D* P2 = &glob_mesh.Points[j];
	point2D* P3 = &glob_mesh.Points[list[0]];
	nor = get_normal(P1,P2,P3);
	double a = dist(P1,P2)/det(P1,P2,P3);
	res.xx = -((P2->y)-(P1->y))*nor.x;
	res.xy = -((P2->y)-(P1->y))*nor.y;
	res.yx = ((P2->x)-(P1->x))*nor.x;
	res.yy = ((P2->x)-(P1->x))*nor.y;
	mat_mult(&res,a/6);
	free(list);
	return res;
}
*/

void insert_B(double* B,double v,int i,int d){
	B[i] = v;
}

void insert_Ba(double* B,double v,int i,int d){
	B[i] = v;
	B[i+d] = v;
}

void insert_A_aV_a(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i,j+d);
}

void insert_pseudo_A_aV_a(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i+d,j+d);
}

void insert_A_bV_ba(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i,j+d);
	insert_sparse(A,v->x,i+d,j+2*d);
	insert_sparse(A,v->y,i+d,j+3*d);
}

void insert_A_abV_b(sparse_matrix* A,matrix2D* v,int i,int j,int d){
	insert_sparse(A,v->xx,i,j);
	insert_sparse(A,v->xy,i,j+d);
	insert_sparse(A,v->yx,i+d,j);
	insert_sparse(A,v->yy,i+d,j+d);
}

void insert_A_baV_b(sparse_matrix* A,matrix2D* v,int i,int j,int d){
	insert_sparse(A,v->xx,i,j);
	insert_sparse(A,v->yx,i,j+d);
	insert_sparse(A,v->xy,i+d,j);
	insert_sparse(A,v->yy,i+d,j+d);
}

void insert_A_bbV_a(sparse_matrix* A,matrix2D* v,int i,int j,int d){
	insert_sparse(A,v->xx+v->yy,i,j);
	insert_sparse(A,v->xx+v->yy,i+d,j+d);
}

void insert_A_abV(sparse_matrix* A,matrix2D* v,int i,int j,int d){
	insert_sparse(A,v->xx,i,j);
	insert_sparse(A,v->xy,i+d,j);
	insert_sparse(A,v->yx,i+2*d,j);
	insert_sparse(A,v->yy,i+3*d,j);
}

void insert_A_bbV(sparse_matrix* A,matrix2D* v,int i,int j,int d){
	insert_sparse(A,v->xx+v->yy,i,j);
}

void insert_A_aV(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i+d,j);
}

void insert_A_aV_b_row_x(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->x,i+d,j+d);
}

void insert_A_aV_b_row_y(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->y,i,j);
	insert_sparse(A,v->y,i+d,j+d);
}

void insert_A_bV_a_row_x(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i+d,j);
}

void insert_A_bV_a_row_y(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j+d);
	insert_sparse(A,v->y,i+d,j+d);
}

void insert_A_cV_cE_ab_row_x(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i,j);
	insert_sparse(A,v->y,i,j+d);
}

void insert_A_cV_cE_ab_row_y(sparse_matrix* A,point2D* v,int i,int j,int d){
	insert_sparse(A,v->x,i+d,j);
	insert_sparse(A,v->y,i+d,j+d);
}

void insert_AV(sparse_matrix* A,double v,int i,int j,int d){
	insert_sparse(A,v,i,j);
}

void insert_AV_a(sparse_matrix* A,double v,int i,int j,int d){
	insert_sparse(A,v,i,j);
	insert_sparse(A,v,i+d,j+d);
}

/*void insert_BV_aW(sparse_matrix* A,double v,double* data,int i,int j,int k,int d){
	insert_sparse(A,data[k]*v,i,j);
	insert_sparse(A,data[k]*v,i+d,j+d);
}

void insert_B_aV_aW(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d){
	vec_mult(v,data[k]);
	insert_A_aV_a(A,v,i,j,d);
}

void insert_B_aVW(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d){
	vec_mult(v,data[k]);
	insert_A_aV(A,v,i,j,d);
}

void insert_B_aVW_a(sparse_matrix* A,point2D* v,double* data,int i,int j,int k,int d){
	double s = (v->x)*data[k]+(v->y)*data[k+d];
	insert_AV(A,s,i,j,d);
}*/

void insert_BWWW(sparse_matrix3D* B,double v,int i,int j,int k,int d){		
	B->Len[i]++;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_BWWW_a(sparse_matrix3D* B,double v,int i,int j,int k,int d){		
	B->Len[i]++;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	B->Len[i+d]++;																	
	len = B->Len[i+d];														
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_aWWW(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){		
	B->Len[i]++;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->x;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	B->Len[i+d]++;																	
	len = B->Len[i+d];														
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k;
}

void insert_B_divergence(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	// only for Bijk_001	
	B->Len[k] += 2;																	
	int len = B->Len[k];														
	B->Values[k] = (double*)realloc(B->Values[k],len*sizeof(double));
	B->Indices1[k] = (int*)realloc(B->Indices1[k],len*sizeof(int));
	B->Indices2[k] = (int*)realloc(B->Indices2[k],len*sizeof(int));
	B->Values[k][len-2] = v->x;
	B->Indices1[k][len-2] = j;
	B->Indices2[k][len-2] = i;
	B->Values[k][len-1] = v->y;
	B->Indices1[k][len-1] = j;
	B->Indices2[k][len-1] = i+d;
}

void insert_B_gradient(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){		// only for Bijk_001	
	B->Len[k]++;																	
	int len = B->Len[k];														
	B->Values[k] = (double*)realloc(B->Values[k],len*sizeof(double));
	B->Indices1[k] = (int*)realloc(B->Indices1[k],len*sizeof(int));
	B->Indices2[k] = (int*)realloc(B->Indices2[k],len*sizeof(int));
	B->Values[k][len-1] = v->x;
	B->Indices1[k][len-1] = j;
	B->Indices2[k][len-1] = i;
	
	B->Len[k+d]++;																	
	len = B->Len[k+d];														
	B->Values[k+d] = (double*)realloc(B->Values[k+d],len*sizeof(double));
	B->Indices1[k+d] = (int*)realloc(B->Indices1[k+d],len*sizeof(int));
	B->Indices2[k+d] = (int*)realloc(B->Indices2[k+d],len*sizeof(int));
	B->Values[k+d][len-1] = v->y;
	B->Indices1[k+d][len-1] = j;
	B->Indices2[k+d][len-1] = i;
}

void insert_BWV_aV_a(sparse_matrix3D* B,double v,int i,int j,int k,int d){		
	B->Len[i] += 2;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
}

void insert_B_bV_bV_a(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){		
	B->Len[i] += 2;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->x;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->y;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k;
	
	B->Len[i+d] += 2;
	len = B->Len[i+d];
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->x;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k;
}

void insert_B_aWWV_a(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){		
	B->Len[i] += 2;																	
	int len = B->Len[i];														
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->x;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->y;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k+d;
}

void insert_B_aaWWW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		
	B->Len[i]++;										
	int len = B->Len[i];						
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->xx+v->yy;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_V_aB_baV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i] += 4;																	//j->V_a
	int len = B->Len[i];															//k->V_b
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-4] = v->xx;
	B->Indices1[i][len-4] = j;
	B->Indices2[i][len-4] = k;
	B->Values[i][len-3] = v->xy;
	B->Indices1[i][len-3] = j+d;
	B->Indices2[i][len-3] = k;
	B->Values[i][len-2] = v->yx;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k+d;
	B->Values[i][len-1] = v->yy;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
}

void insert_V_aB_abV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i] += 4;																	//j->V_b
	int len = B->Len[i];															//k->V_a
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-4] = v->xx;
	B->Indices1[i][len-4] = j;
	B->Indices2[i][len-4] = k;
	B->Values[i][len-3] = v->yx;
	B->Indices1[i][len-3] = j+d;
	B->Indices2[i][len-3] = k;
	B->Values[i][len-2] = v->xy;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k+d;
	B->Values[i][len-1] = v->yy;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
}

void insert_UBa_V_a(sparse_matrix3D*B,point2D* v,int i,int j,int k,int d){				//i->W
	B->Len[i] += 2;																	//j->U
	int len = B->Len[i];															//k->V_a
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->x;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->y;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k+d;
	
}

void insert_B_aaV_bV_b(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){     //i-> W
	B->Len[i] += 2;																	//j->V_b
	int len = B->Len[i];															//k->V_b
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->xx+v->yy;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->xx+v->yy;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
}

void insert_WB_xx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i]++;																	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->xx;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_WB_xy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i]++;																	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->xy;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_WB_yx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i]++;																	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->yx;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_WB_yy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){		//i-> W
	B->Len[i]++;																	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->yy;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
}

void insert_B_abV_cV_c_rowx(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){     //i-> W
	int len;
	B->Len[i] += 2;																		//j->V_c
	len = B->Len[i];																	//k->V_c
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->xx;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->xx;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
	
	B->Len[i+d] += 2;																
	len = B->Len[i+d];														
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->xy;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->xy;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_abV_cV_c_rowy(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){    //i-> W
	int len;
	B->Len[i] += 2;																		//j->V_c
	len = B->Len[i];																	//k->V_c
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->yx;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->yx;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k+d;
	
	B->Len[i+d] += 2;																
	len = B->Len[i+d];														
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->yy;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->yy;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_aaV_bV_bW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){	//i-> V_b
	(B->Len[i])++;																	//j-> V_b
	(B->Len[i+d])++;																//k-> W
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->xx+v->yy;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];		
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->xx+v->yy;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k;
}

void insert_B_abV_bV_aW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){	//i-> V_b
	(B->Len[i]) += 2;																//j-> V_a
	(B->Len[i+d]) += 2;																//k-> W
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->xx;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->xy;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->yx;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->yy;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k;
}

void insert_B_baV_bV_aW(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){	//i-> V_a
	(B->Len[i]) += 2;																//j-> V_b
	(B->Len[i+d]) += 2;																//k-> W
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->xx;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->yx;
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->xy;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->yy;
	B->Indices1[i+d][len-1] = j+d;
	B->Indices2[i+d][len-1] = k;
	
}

void insert_B_aWV_b_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> B_a
	(B->Len[i])++;																	//j-> W
	(B->Len[i+d])++;																//k-> V_b
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->x;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->x;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_aWV_b_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> B_a
	(B->Len[i])++;																	//j-> W
	(B->Len[i+d])++;																//k-> V_b
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->y;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_bWV_a_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> B_b
	(B->Len[i])++;																	//j-> W
	(B->Len[i+d])++;																//k-> V_a
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->x;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k;
}

void insert_B_bWV_a_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> B_b
	(B->Len[i])++;																	//j-> W
	(B->Len[i+d])++;																//k-> V_a
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-1] = v->x;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k+d;
	
	len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_B_cWV_cEab_rowx(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> B_c
	(B->Len[i]) += 2;																	//j-> W
																					//k-> V_c
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = v->x;
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = v->y;
	B->Indices1[i][len-1] = j;
	B->Indices2[i][len-1] = k+d;
}

void insert_B_cWV_cEab_rowy(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d){	//i-> A_a
																					//j-> W
	(B->Len[i+d]) += 2;																//k-> V_b
	
	int len = B->Len[i+d];															
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = v->x;
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = v->y;
	B->Indices1[i+d][len-1] = j;
	B->Indices2[i+d][len-1] = k+d;
}

void insert_WWV_aC_baV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d){
	sparse_general_append(A,v->xx,4,i,j,k,l);
	sparse_general_append(A,v->xy,4,i,j,k,l+d);
	sparse_general_append(A,v->yx,4,i,j,k+d,l);
	sparse_general_append(A,v->yy,4,i,j,k+d,l+d);
}

void insert_WWC_aaV_bV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d){
	sparse_general_append(A,v->xx+v->yy,4,i,j,k,l);
	sparse_general_append(A,v->xx+v->yy,4,i,j,k+d,l+d);
}

void insert_WWV_aC_abV_b(sparse_general* A,matrix2D* v,int i,int j,int k,int l,int d){
	sparse_general_append(A,v->xx,4,i,j,k,l);
	sparse_general_append(A,v->yx,4,i,j,k,l+d);
	sparse_general_append(A,v->xy,4,i,j,k+d,l);
	sparse_general_append(A,v->yy,4,i,j,k+d,l+d);
}

/*(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d){
	double res = v->xx*X2[k]*X3[l]+v->xy*X2[k+d]*X3[l]+v->yx*X2[k]*X3[l+d]+v->yy*X2[k+d]*X3[l+d];
	V[i] = res*X1[j];
}

void insert_WV_abV_ab(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d){
	double res = (v->xx+v->yy)*(X2[k]*X3[l]+X2[k+d]*X3[l+d]);
	V[i] = res*X1[j];
}

void insert_WV_aaV_bb(double* V,matrix2D* v,double* X1,double* X2,double* X3,int i,int j,int k,int l,int d){
	double res = v->xx*X2[k]*X3[l]+v->xy*X2[k]*X3[l+d]+v->yx*X2[k+d]*X3[l]+v->yy*X2[k+d]*X3[l+d];
	V[i] = res*X1[j];
}*/

double* set_power_0(double (*func)(point2D* P),int equ){
	double sum;
	int i,j,m,index_i;
	int d = glob_mesh.size;
	double* Res = (double*)malloc(var_number_2D*d*sizeof(double));
	for (i=0;i<var_number_2D*d;i++){Res[i] = 0;}
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		sum = matrix_Aii_00(i)*(*func)(&glob_mesh.Points[i]);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			sum += matrix_Aij_00(i,j)*(*func)(&glob_mesh.Points[j]);
		}
		Res[index_i] = sum;
	}
	return Res;
}

double* set_power_1(double* Solution,int equ,int var,double factor){
	int i,j,m,index_i,index_j;
	double sum;
	int d = glob_mesh.size;
	double* Res = (double*)malloc(var_number_2D*d*sizeof(double));
	for (i=0;i<var_number_2D*d;i++){Res[i] = 0;}
	for (i=0;i<d;i++){
		index_i = d*var+i;
		sum = matrix_Aii_00(i)*Solution[index_i];
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			sum += matrix_Aij_00(i,j)*Solution[index_j];
		}
		Res[d*equ+i] = factor*sum;
	}
	return Res;
}

/*double* set_power_2(double* Solution,int equ,int var,double factor){
	int i,j,k,m,index_i,index_j,index_k;
	double sum;
	int d = glob_mesh.size;
	int* list;
	double* Res = (double*)malloc(var_number_2D*d*sizeof(double));
	for (i=0;i<var_number_2D*d;i++){Res[i] = 0;}
	for (i=0;i<d;i++){
		index_i = d*var+i;
		sum = matrix_Biii_000(i)*Solution[index_i]*Solution[index_i];
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			sum += 2*matrix_Biij_000(i,j)*Solution[index_i]*Solution[index_j];
			list = overlap(i,j);
			k = list[0];
			index_k = d*var+k;
			sum += matrix_Bijk_000(i,j,k)*Solution[index_j]*Solution[index_k];
			if (list[1]>=0){
				k = list[1];
				index_k = d*var+k;
				sum += matrix_Bijk_000(i,j,k)*Solution[index_j]*Solution[index_k];
			}
			free(list);
		}
		Res[d*equ+i] = factor*sum;
	}
	return Res;
}

double* set_power_3(double* Solution,int equ,int var,double factor){
	int i,j,k,m,index_i,index_j,index_k;
	double sum;
	int d = glob_mesh.size;
	int* list;
	double* Res = (double*)malloc(var_number_2D*d*sizeof(double));
	for (i=0;i<var_number_2D*d;i++){Res[i] = 0;}
	for (i=0;i<d;i++){
		index_i = d*var+i;
		sum = matrix_Ciiii_0000(i)*Solution[index_i]*Solution[index_i]*Solution[index_i];
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			sum += 3*matrix_Ciiij_0000(i,j)*Solution[index_i]*Solution[index_i]*Solution[index_j];
			sum += 3*matrix_Ciijj_0000(i,j)*Solution[index_i]*Solution[index_j]*Solution[index_j];
			list = overlap(i,j);
			k = list[0];
			index_k = d*var+k;
			sum += 5*matrix_Cijkk_0000(i,j,k)*Solution[index_i]*Solution[index_j]*Solution[index_k];
			if (list[1]>=0){
				k = list[1];
				index_k = d*var+k;
				sum += 5*matrix_Cijkk_0000(i,j,k)*Solution[index_i]*Solution[index_j]*Solution[index_k];
			}
			free(list);
		}
		Res[d*equ+i] = factor*sum;
	}
	return Res;
}*/

double get_total_area(){
	int i;
	int d = glob_mesh.size;
	double* U = set_vector_Ui_0_2D(1);
	double sum = 0;
	for (i=0;i<d;i++) sum += U[i];
	free(U);
	return sum;
}

double get_total_amount(double* X,int index){
	int i;
	int d = glob_mesh.size;
	double* U = set_vector_Ui_0_2D(1);
	double sum = 0;
	//for (i=0;i<d;i++) sum += X[i+index*d]*Dual_areas[i];
	for (i=0;i<d;i++) sum += X[i+index*d]*U[i];
	free(U);
	return sum;
}

sparse_general* set_matrix_Cijkl_0011_2D(int var1,int var2,int var3,int var4,void (*insert)(sparse_general* V,matrix2D* v,int i,int j,int k,int l,int d)){
	int I,J,i,j,k,x1,y1,z2;
	int n = glob_mesh.size;
	double detF;
	int* list;
	point2D grad1;
	point2D grad2;
	matrix2D* A;
	matrix2D* B;
	
	int* Map0[3] = {&i,&j,&k};
	int* Map1[3] = {&k,&i,&j};
	int* Map2[3] = {&j,&k,&i};
	int* Map3[3] = {&j,&i,&k};
	int* Map4[3] = {&i,&k,&j};
	int* Map5[3] = {&k,&j,&i};
	
	int*** Combi_map = (int***)malloc(6*sizeof(int**));
	Combi_map[0] = Map0;
	Combi_map[1] = Map1;
	Combi_map[2] = Map2;
	Combi_map[3] = Map3;
	Combi_map[4] = Map4;
	Combi_map[5] = Map5;
	
	int* Dim = zero_int_list(4);
	Dim[0] = n;
	Dim[1] = n;
	Dim[2] = 2*n;
	Dim[3] = 2*n;
	sparse_general* Res = new_sparse_general(4,Dim);	
	free(Dim);		
	for (I=0;I<elements.size;I++){
		i = elements.Elements[I].i;
		j = elements.Elements[I].j;
		k = elements.Elements[I].k;
		detF = fabs(det_tri(i,j,k));
		
		grad1 = grad_2D(i,j,k);							// xxxx				1 in 4
		A = tensor_product(&grad1,&grad1);
		mat_mult(A,detF/12.);
		(*insert)(Res,A,var1*n+i,var2*n+i,var3*n+i,var4*n+i,n);
		free(A);
		
		grad1 = grad_2D(j,k,i);							
		A = tensor_product(&grad1,&grad1);
		mat_mult(A,detF/12.);
		(*insert)(Res,A,var1*n+j,var2*n+j,var3*n+j,var4*n+j,n);
		free(A);
		
		grad1 = grad_2D(k,i,j);				
		A = tensor_product(&grad1,&grad1);
		mat_mult(A,detF/12.);
		(*insert)(Res,A,var1*n+k,var2*n+k,var3*n+k,var4*n+k,n);
		free(A);
		
		for (J=0;J<6;J++){
			x1 = *Combi_map[J][0];
			y1 = *Combi_map[J][1];
			z2 = *Combi_map[J][2];
			
			grad1 = grad_2D(x1,y1,z2);					// zzxy				3 in 4
			grad2 = grad_2D(y1,x1,z2);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/12.);
			(*insert)(Res,A,var1*n+z2,var2*n+z2,var3*n+x1,var4*n+y1,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// xzzy
			grad2 = grad_2D(y1,x1,z2);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+x1,var2*n+z2,var3*n+z2,var4*n+y1,n);
			free(A);
			
			grad1 = grad_2D(z2,x1,y1);					// xyzz		
			A = tensor_product(&grad1,&grad1);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+x1,var2*n+y1,var3*n+z2,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(y1,z2,x1);					// zxyz
			grad2 = grad_2D(z2,x1,y1);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+z2,var2*n+x1,var3*n+y1,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// zxzy
			grad2 = grad_2D(y1,x1,z2);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+z2,var2*n+x1,var3*n+z2,var4*n+y1,n);
			free(A);
			
			grad1 = grad_2D(y1,z2,x1);					// xzyz
			grad2 = grad_2D(z2,x1,y1);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+x1,var2*n+z2,var3*n+y1,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(x1,y1,z2);					// zzxx				2 in 4
			A = tensor_product(&grad1,&grad1);
			mat_mult(A,detF/12.);
			(*insert)(Res,A,var1*n+z2,var2*n+z2,var3*n+x1,var4*n+x1,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// zxzx
			grad2 = grad_2D(x1,z2,y1);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+z2,var2*n+x1,var3*n+z2,var4*n+x1,n);
			free(A);
			
			grad1 = grad_2D(x1,y1,z2);					// zxxz
			grad2 = grad_2D(z2,x1,y1);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+z2,var2*n+x1,var3*n+x1,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// xzzz
			A = tensor_product(&grad1,&grad1);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+x1,var2*n+z2,var3*n+z2,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// zxzz
			A = tensor_product(&grad1,&grad1);
			mat_mult(A,detF/24.);
			(*insert)(Res,A,var1*n+z2,var2*n+x1,var3*n+z2,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(x1,y1,z2);					// zzxz
			grad2 = grad_2D(z2,x1,y1);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/12.);
			(*insert)(Res,A,var1*n+z2,var2*n+z2,var3*n+x1,var4*n+z2,n);
			free(A);
			
			grad1 = grad_2D(z2,y1,x1);					// zzzx
			grad2 = grad_2D(x1,y1,z2);
			A = tensor_product(&grad1,&grad2);
			mat_mult(A,detF/12.);
			(*insert)(Res,A,var1*n+z2,var2*n+z2,var3*n+z2,var4*n+x1,n);
			free(A);
		}
	}
	return Res;
}

/*sparse_general* set_matrix_Cijkl_0011_2D(int equ,int var1,int var2,int var3,void (*insert)(sparse_general* V,matrix2D* v,int i,int j,int k,int l,int d)){
	int i,j,k,m,index_i,index_j,index_k,index_l;
	const int rank = 4;
	double a;
	int* list;
	int d = glob_mesh.size;
	int* Dim = zero_int_list(4);
	Dim[0] = d;
	Dim[1] = d;
	Dim[2] = var_number_2D*d;
	Dim[3] = var_number_2D*d;
	sparse_general* Res = new_sparse_general(rank,Dim);						 			
	for (i=0;i<d;i++){														
		matrix2D v;
		index_i = d*equ+i;
		index_j = d*var1+i;
		index_k = d*var2+i;
		index_l = d*var3+i;
		v = matrix_Ciiii_0011(i);						// Ciiii
		(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var1+i;							// Ciiij
			index_k = d*var2+i;
			index_l = d*var3+j;
			v = matrix_Ciiij_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+i;							// Ciiji
			index_k = d*var2+j;
			index_l = d*var3+i;
			v = matrix_Ciiji_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+j;							// Cijii
			index_k = d*var2+i;
			index_l = d*var3+i;
			v = matrix_Cijii_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+j;							// Cijjj
			index_k = d*var2+j;
			index_l = d*var3+j;
			v = matrix_Cijii_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+i;							// Ciijj
			index_k = d*var2+j;
			index_l = d*var3+j;
			v = matrix_Ciijj_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+j;							// Cijij
			index_k = d*var2+i;
			index_l = d*var3+j;
			v = matrix_Cijij_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
			
			index_j = d*var1+j;							// Cijji
			index_k = d*var2+j;
			index_l = d*var3+i;
			v = matrix_Cijji_0011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);
														
			list = overlap(i,j);		
			k = list[0];
							
			index_j = d*var1+j;							// Cijkk
			index_k = d*var2+k;		
			index_l = d*var3+k;
			v = matrix_Cijkk_0011(i,j,k);			
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);			
			
			index_j = d*var1+j;							// Cijjk
			index_k = d*var2+j;		
			index_l = d*var3+k;
			v = matrix_Cijjk_0011(i,j,k);			
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
			
			index_j = d*var1+i;							// Ciijk
			index_k = d*var2+j;		
			index_l = d*var3+k;
			v = matrix_Ciijk_0011(i,j,k);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
			
			index_j = d*var1+j;							// Cijik
			index_k = d*var2+i;		
			index_l = d*var3+k;
			v = matrix_Cijik_0011(i,j,k);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
			
			index_j = d*var1+j;							// Cijki
			index_k = d*var2+k;		
			index_l = d*var3+i;
			v = matrix_Cijki_0011(i,j,k);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
			
			index_j = d*var1+j;							// Cijkj
			index_k = d*var2+k;		
			index_l = d*var3+j;
			v = matrix_Cijkj_0011(i,j,k);
			(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
			
			if (list[1]>=0){
				k = list[1];
				
				index_j = d*var1+j;							// Cijkk
				index_k = d*var2+k;		
				index_l = d*var3+k;
				v = matrix_Cijkk_0011(i,j,k);			
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);			
			
				index_j = d*var1+j;							// Cijjk
				index_k = d*var2+j;		
				index_l = d*var3+k;
				v = matrix_Cijjk_0011(i,j,k);			
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
				
				index_j = d*var1+i;							// Ciijk
				index_k = d*var2+j;		
				index_l = d*var3+k;
				v = matrix_Ciijk_0011(i,j,k);
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
				
				index_j = d*var1+j;							// Cijik
				index_k = d*var2+i;		
				index_l = d*var3+k;
				v = matrix_Cijik_0011(i,j,k);
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
				
				index_j = d*var1+j;							// Cijki
				index_k = d*var2+k;		
				index_l = d*var3+i;
				v = matrix_Cijki_0011(i,j,k);
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);	
				
				index_j = d*var1+j;							// Cijkj
				index_k = d*var2+k;		
				index_l = d*var3+j;
				v = matrix_Cijkj_0011(i,j,k);
				(*insert)(Res,&v,index_i,index_j,index_k,index_l,d);				
			}
			free(list);			
		}
	}
	return Res;
}*/

double* set_vector_Ui_0_2D(int var_num){
	int i,j;
	double u;
	int d = glob_mesh.size;
	double* Res = zero_vector(d*var_num);
	for (i=0;i<d;i++){
		u = vector_Ui_0(i);
		for (j=0;j<var_num;j++) Res[d*j+i] = u;
	 }
	return Res;
}

double* Get_vector_Ui_0_2D(mesh2D* Mesh){
	int i,j;
	double u;
	int d = Mesh->size;
	double* Res = zero_vector(d);
	for (i=0;i<d;i++) Res[i] = get_vector_Ui_0(Mesh,i);
	return Res;
}

double* set_vector_bi_0(int var,void (*insert)(double* B,double v,int i,int d)){
	int i,index_i;	
	int d = glob_mesh.size;
	double v;
	double* Res = zero_vector(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*var+i;
		v = vector_b_0(i);
		(*insert)(Res,v,index_i,d);
	}
	return Res;
}

double* glob_div(int var){		// Achtung gilt nur für zusammenhängende Ränder !
	point2D n;
	int i,start,old,novel,inner;
	double v;
	int d = glob_mesh.size;
	double* Res = zero_vector(var_number_2D*d);
	start = -1;
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		start = i;
		break;
	}
	old = -1;
	i = start;
	do{
		novel = glob_mesh.Connections[i][0];
		inner = glob_mesh.Connections[i][1];
		if (novel==old){
			novel = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
			inner = glob_mesh.Connections[i][glob_mesh.Sizes[i]-2];
		}
		v = vector_b_0(i);
		n = get_normal(&glob_mesh.Points[novel],&glob_mesh.Points[i],&glob_mesh.Points[inner]);
		Res[d*var+i] = v*n.x;
		Res[d*(var+1)+i] = v*n.y;
		old = i;
		i = novel;
	}while(novel!=start);
	return Res;
}

double* glob_rot(int var){		// Achtung gilt nur für zusammenhängende Ränder !
	point2D* n;
	int i,start,old,novel,inner;
	double v;
	int d = glob_mesh.size;
	double* Res = zero_vector(var_number_2D*d);
	start = -1;
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		start = i;
		break;
	}
	old = -1;
	i = start;
	do{
		novel = glob_mesh.Connections[i][0];
		inner = glob_mesh.Connections[i][1];
		if (novel==old){
			novel = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
			inner = glob_mesh.Connections[i][glob_mesh.Sizes[i]-2];
		}
		v = vector_b_0(i);
		n = clone_point(&glob_mesh.Points[novel]);
		vec_add_mult(n,&glob_mesh.Points[i],-1.);
		normalize(n);
		Res[d*var+i] = v*(n->x);
		Res[d*(var+1)+i] = v*(n->y);
		old = i;
		i = novel;
		free(n);
	}while(novel!=start);
	return Res;
}

sparse_matrix* set_matrix_Aij_00_2D(int equ,int var,
void (*insert)(sparse_matrix* A,double v,int i,int j,int d)){
	int i,j,m,index_i,index_j;
	int d = glob_mesh.size;
	double v;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_Aii_00(i);
		(*insert)(Res,v,index_i,index_j,d);	
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			v = matrix_Aij_00(i,j);
			(*insert)(Res,v,index_i,index_j,d);		
		}
	}
	return Res;
}

sparse_matrix* get_matrix_Aij_00_2D(mesh2D* Mesh,int equ,int var,
void (*insert)(sparse_matrix* A,double v,int i,int j,int d)){
	int i,j,m,index_i,index_j;
	int d = Mesh->size;
	double v;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = get_matrix_Aii_00(Mesh,i);
		(*insert)(Res,v,index_i,index_j,d);		
		for (m=0;m<Mesh->Sizes[i];m++){
			j = Mesh->Connections[i][m];
			index_j = d*var+j;
			v = get_matrix_Aij_00(Mesh,i,j);
			(*insert)(Res,v,index_i,index_j,d);			
		}
	}
	return Res;
}

sparse_matrix* set_matrix_Aij_01_2D(int equ,int var,
void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d)){
	int i,j,m,index_i,index_j;
	int d = glob_mesh.size;
	point2D v;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_Aii_01(i);
		(*insert)(Res,&v,index_i,index_j,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			v = matrix_Aij_01(i,j);
			(*insert)(Res,&v,index_i,index_j,d);
		}
	}
	return Res;
}

sparse_matrix* set_matrix_Aij_10_2D(int equ,int var,
void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d)){
	int i,j,m,index_i,index_j;
	int d = glob_mesh.size;
	point2D v;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_Aii_01(i);
		(*insert)(Res,&v,index_i,index_j,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			v = matrix_Aij_01(j,i);
			(*insert)(Res,&v,index_i,index_j,d);
		}
	}
	return Res;
}

sparse_matrix* set_matrix_Aij_11_2D(int equ,int var,
void (*insert)(sparse_matrix* A,matrix2D* v,int i,int j,int d)){
	int i,j,m,index_i,index_j;
	int d = glob_mesh.size;
	matrix2D v;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_Aii_11(i);
		(*insert)(Res,&v,index_i,index_j,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var+j;
			v = matrix_Aij_11(i,j);
			(*insert)(Res,&v,index_i,index_j,d);
		}
	}
	return Res;
}

/*sparse_matrix* set_matrix_bound_Aij_00_2D(int** conditions,int cond,int equ,int var,
void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d)){
	point2D v;
	int i,j,index_i,index_j;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		if(conditions[i][equ]==cond){
			index_i = d*equ+i;
			index_j = d*var+i;
			v = matrix_bound_1_Aii_00(i);
			(*insert)(Res,&v,index_i,index_j,d);
			j = glob_mesh.Connections[i][0];
			index_j = d*var+j;
			v = matrix_bound_1_Aij_00(i,j);
			(*insert)(Res,&v,index_i,index_j,d);
			j = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
			index_j = d*var+j;
			v = matrix_bound_1_Aij_00(i,j);
			(*insert)(Res,&v,index_i,index_j,d);
		}
	}
	return Res;
}*/

/*sparse_matrix* set_matrix_bound_0_Aij_00_2D(int equ,int var,void (*insert)(sparse_matrix* A,double v,int i,int j,int d)){
	int i,j,k,index_i,index_j;
	double v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++) if (is_boundary(i)>=0){
		index_i = d*equ+i;
		index_j = d*var+i;
		v = matrix_bound_0_Aii_00(i);
		(*insert)(Res,v,index_i,index_j,d);
		for (k=0;k<glob_mesh.Sizes[i];k++){
			j = glob_mesh.Connections[i][k];
			if (is_boundary(j)>=0){
				index_j = d*var+j;
				v = matrix_bound_0_Aij_00(i,j);
				(*insert)(Res,v,index_i,index_j,d);
			}
		}
	}
	return Res;
}


sparse_matrix* set_matrix_bound_1_Aij_00_2D(int** conditions,int cond,int equ,int var,
void (*insert)(sparse_matrix* A,point2D* v,int i,int j,int d)){
	point2D v;
	int i,j,k,index_i,index_j,b1,b2;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		b1 = (conditions!=NULL && conditions[i][equ]==cond);
		b2 = (partition_mode && Partition[i]==0);
		if(b1 || b2){
			index_i = d*equ+i;
			index_j = d*var+i;
			v = matrix_bound_1_Aii_00(i);
			(*insert)(Res,&v,index_i,index_j,d);
			for (k=0;k<glob_mesh.Sizes[i];k++){
				j = glob_mesh.Connections[i][k];
				index_j = d*var+j;
				v = matrix_bound_1_Aij_00(i,j);
				(*insert)(Res,&v,index_i,index_j,d);
			}
		}
	}
	return Res;
}

sparse_matrix* set_matrix_bound_0_Aij_01_2D(int** conditions,int cond,int equ,int var,
void (*insert)(sparse_matrix* A,double v,int i,int j,int d)){
	int i,j,k,index_i,index_j,b1,b2;
	double v;
	int d = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(var_number_2D*d);
	for (i=0;i<d;i++){
		b1 = (conditions!=NULL && conditions[i][equ]==cond);
		b2 = (partition_mode && Partition[i]==0);
		if (b1 || b2){
			index_i = d*equ+i;
			index_j = d*var+i;
			v = matrix_bound_0_Aii_01(i);
			(*insert)(Res,v,index_i,index_j,d);
			for (k=0;k<glob_mesh.Sizes[i];k++){
				j = glob_mesh.Connections[i][k];
				index_j = d*var+j;
				v = matrix_bound_0_Aij_01(i,j);
				(*insert)(Res,v,index_i,index_j,d);				
			}			
		}
	}
	return Res;
}*/

sparse_matrix3D* set_matrix_Bijk_011(int equ,int var1,int var2,
  void (*insert)(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d)){
	int i,j,k,m,index_i,index_j,index_k;
	int* list;
	int d = glob_mesh.size;
	sparse_matrix3D* Res = sparse_zero3D(d*var_number_2D);						 			
	for (i=0;i<d;i++){														
		matrix2D v;
		index_i = d*equ+i;
		index_j = d*var1+i;
		index_k = d*var2+i;
		v = matrix_Biii_011(i);							// Biii
		(*insert)(Res,&v,index_i,index_j,index_k,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var1+i;							// Biij
			index_k = d*var2+j;
			v = matrix_Biij_011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Biji
			index_k = d*var2+i;
			v = matrix_Biji_011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Bijj
			index_k = d*var2+j;
			v = matrix_Bijj_011(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			list = overlap(i,j);						// Bijk
			index_j = d*var1+j;
			index_k = d*var2+list[0];		
			v = matrix_Bijk_011(i,j,list[0]);			
			(*insert)(Res,&v,index_i,index_j,index_k,d);			
			if (list[1]>=0){
				index_k = d*var2+list[1];		
				v = matrix_Bijk_011(i,j,list[1]);
				(*insert)(Res,&v,index_i,index_j,index_k,d);
			}
			free(list);
		}
	}
	return Res;
}

sparse_matrix3D* set_matrix_Bijk_000(int equ,int var1,int var2,				
  void (*insert)(sparse_matrix3D* B,double v,int i,int j,int k,int d)){
	int i,j,k,m,index_i,index_j,index_k;
	int* list;
	int d = glob_mesh.size;
	sparse_matrix3D* Res = sparse_zero3D(d*var_number_2D);						 			
	for (i=0;i<d;i++){														
		double v;
		index_i = d*equ+i;
		index_j = d*var1+i;
		index_k = d*var2+i;
		v = matrix_Biii_000(i);							// Biii
		(*insert)(Res,v,index_i,index_j,index_k,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var1+i;							// Biij
			index_k = d*var2+j;
			v = matrix_Biij_000(i,j);
			(*insert)(Res,v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Biji
			index_k = d*var2+i;
			v = matrix_Biji_000(i,j);
			(*insert)(Res,v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Bijj
			index_k = d*var2+j;
			v = matrix_Bijj_000(i,j);
			(*insert)(Res,v,index_i,index_j,index_k,d);
			
			list = overlap(i,j);						// Bijk
			index_j = d*var1+j;
			index_k = d*var2+list[0];		
			v = matrix_Bijk_000(i,j,list[0]);			
			(*insert)(Res,v,index_i,index_j,index_k,d);			
			if (list[1]>=0){
				index_k = d*var2+list[1];		
				v = matrix_Bijk_000(i,j,list[1]);
				(*insert)(Res,v,index_i,index_j,index_k,d);
			}
			free(list);
		}
	}
	return Res;
}

sparse_matrix3D* set_matrix_Bijk_001(int equ,int var1,int var2,					// first index test func, second damage field, third shift field
  void (*insert)(sparse_matrix3D* B,point2D* v,int i,int j,int k,int d)){
	int i,j,k,m,index_i,index_j,index_k;
	int* list;
	int d = glob_mesh.size;
	sparse_matrix3D* Res = sparse_zero3D(d*var_number_2D);						 			
	for (i=0;i<d;i++){														
		point2D v;
		index_i = d*equ+i;
		index_j = d*var1+i;
		index_k = d*var2+i;
		v = matrix_Biii_001(i);							// Biii
		(*insert)(Res,&v,index_i,index_j,index_k,d);
		for (m=0;m<glob_mesh.Sizes[i];m++){
			j = glob_mesh.Connections[i][m];
			index_j = d*var1+i;							// Biij
			index_k = d*var2+j;
			v = matrix_Biij_001(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Biji
			index_k = d*var2+i;
			v = matrix_Biji_001(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			index_j = d*var1+j;							// Bijj
			index_k = d*var2+j;
			v = matrix_Bijj_001(i,j);
			(*insert)(Res,&v,index_i,index_j,index_k,d);
			
			list = overlap(i,j);						// Bijk
			index_j = d*var1+j;
			index_k = d*var2+list[0];		
			v = matrix_Bijk_001(i,j,list[0]);			
			(*insert)(Res,&v,index_i,index_j,index_k,d);			
			if (list[1]>=0){
				index_k = d*var2+list[1];		
				v = matrix_Bijk_001(i,j,list[1]);
				(*insert)(Res,&v,index_i,index_j,index_k,d);
			}
			free(list);
		}
	}
	return Res;
}

sparse_matrix* set_matrix_unity(int equ,int var){
	int i;
	int n = glob_mesh.size*var_number_2D;
	sparse_matrix* Res = sparse_zero(n);
	for (i=0;i<glob_mesh.size;i++) insert_sparse(Res,1.,i+equ*glob_mesh.size,i+var*glob_mesh.size);
	return Res;
}

double cos_2D(int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	if (x==0) return 0;
	else{
		double a = y/x;
		if (x<0) return -1./sqrt(1+a*a);
		else return 1./sqrt(1+a*a);
	}
}

double sin_2D(int i){
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	if (y==0) return 0;
	else{
		double a = y/x;
		if (y<0) return -fabs(a)/sqrt(1+a*a);
		else return fabs(a)/sqrt(1+a*a);
	}
}

double arc_cos(double x){
	return atan(sqrt(1.-x*x)/x);
}

double arc_sin(double x){
	return atan(x/sqrt(1.-x*x));
}

double Tschebyscheff_disc(int i,int n,int m,int trig){
	double res;
	double x = glob_mesh.Points[i].x;
	double y = glob_mesh.Points[i].y;
	double r = sqrt(x*x+y*y);
	double phi = atan2(y,x);
	if (trig==0) res = cos((double)m*phi); else res = sin((double)m*phi);
	res *= cos((double)n*arc_cos(r))/(2.*M_PI*M_PI*sqrt(1-r*r));
	if (n==0) res *= 2.;
	if (m==0) res *= 2.;
	return res;
}

int FEM2D_rb_comp_ind(const void* A,const void* B){
	int* a = (int*)A;
	int* b = (int*)B;
	if ((*a)<(*b)) return 1; else return 0;
}

void FEM2D_rb_print_key(const void* A){
	int* a = (int*)A;
	printf("%d  |",*a);
}

void FEM2D_rb_print_info(void* I){
	int* i = (int*)I;
	printf("%d\n",*i);
}

void FEM2D_rb_free_key(void* I){
	int* i = (int*)I;
	free(i);
}

void FEM2D_rb_free_info(void* I){
	int* i = (int*)I;
	free(i);
}

int* get_optimized_index_map(){
	int* Index;
	int* Value;
	rb_red_blk_node* Smallest;
	int i,j,k,Ind;
	int n = glob_mesh.size;
	int* Map = (int*)malloc(n*sizeof(int));
	for (i=0;i<n;i++) Map[i] = -1;
	rb_red_blk_tree* Index_tree = RBTreeCreate(&FEM2D_rb_comp_ind,&FEM2D_rb_free_key,
	 &FEM2D_rb_free_info,&FEM2D_rb_print_key,&FEM2D_rb_print_info);
	Index = (int*)malloc(sizeof(int));
	Value = (int*)malloc(sizeof(int));
	*Index = 0;
	*Value = 0;
	Map[0] = 0;
	RBTreeInsert(Index_tree,Value,Index);
	Smallest = RBLargest(Index_tree);
	i = 1;
	do{
		Ind = *(int*)Smallest->info;
		for (j=0;j<glob_mesh.Sizes[Ind];j++){
			k = glob_mesh.Connections[Ind][j];
			if (Map[k]<0){
				Index = (int*)malloc(sizeof(int));
				Value = (int*)malloc(sizeof(int));
				*Index = k;
				*Value = i;
				RBTreeInsert(Index_tree,Value,Index);
				Map[k] = i;
				i++;
			}
		}
		RBDelete(Index_tree,Smallest);
		Smallest = RBLargest(Index_tree);
	}while(Smallest!=NULL);
	RBTreeDestroy(Index_tree);
	return Map;
}

void FEM2D_get_optimized_index_map(int* Map,int* Inv_Map,int varnum){
	int i,j;
	int n = glob_mesh.size;
	int* Map1D = get_optimized_index_map();
	for (i=0;i<n;i++){
		for (j=0;j<varnum;j++){
			if (Map!=NULL) Map[i+j*n] = Map1D[i]*varnum+j;
			if (Inv_Map!=NULL) Inv_Map[Map1D[i]*varnum+j] = i+j*n;
		}
	}
	free(Map1D);
}

int get_nearest_index(point2D* P){
	int i;
	double d;
	int min = 0;
	double dmin = dist(P,&(glob_mesh.Points[0]));
	for (i=1;i<glob_mesh.size;i++){
		d = dist(P,&(glob_mesh.Points[i]));
		if (d<dmin){
			min = i;
			dmin = d;
		}
	}
	return min;
}

int Get_nearest_index(mesh2D* Mesh,point2D* P){
	int i;
	double d;
	int min = 0;
	double dmin = dist(P,&(Mesh->Points[0]));
	for (i=1;i<Mesh->size;i++){
		d = dist(P,&(Mesh->Points[i]));
		if (d<dmin){
			min = i;
			dmin = d;
		}
	}
	return min;
}

int get_nearest_triangle(mesh2D* Mesh,element_collection* Triangles,point2D* P){
	int i;
	double d;
	index3D ind;
	point2D c;
	point2D* A;
	point2D* B;
	point2D* C;
		
	int m = Triangles->size;
	int min = -1;
	double dmin = DBL_MAX;
	for (i=0;i<m;i++){
		ind = Triangles->Elements[i];
		A = &(Mesh->Points[ind.i]);
		B = &(Mesh->Points[ind.j]);
		C = &(Mesh->Points[ind.k]);
		c = triangle_center(A,B,C);
		d = dist(P,&c);
		if (d<dmin){
			min = i;
			dmin = d;
		}
	}
	return min;
}

point2D* get_mesh_normal(int i){
	int s;
	int* List;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	point2D* Res;
	point2D n1,n2;
	P3 = &glob_mesh.Points[i];
	if (is_boundary(i)>=0){
		s = glob_mesh.Connections[i][0];
		P2 = &glob_mesh.Points[s];
		List = overlap(i,s);
		P1 = &glob_mesh.Points[List[0]];
		n1 = get_normal(P3,P2,P1);
		vec_mult(&n1,dist(P3,P2));
		free(List);
		s = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
		P2 = &glob_mesh.Points[s];
		List = overlap(i,s);
		P1 = &glob_mesh.Points[List[0]];
		n2 = get_normal(P3,P2,P1);
		vec_mult(&n2,dist(P3,P2));
		free(List);
		
		Res = clone_point(&n1);
		vec_add_mult(Res,&n2,1.);
		normalize(Res);
		return Res;
	}
	else return NULL;
}

/*double get_bound_curvature(int i){
	int s;
	int* List;
	double K,l1,l2,l;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	point2D* R1;
	point2D* R2;
	point2D n1,n2;
	P3 = &glob_mesh.Points[i];
	if (is_boundary(i)>=0){
		s = glob_mesh.Connections[i][0];
		P2 = &glob_mesh.Points[s];
		l2 = dist(P3,P2);
		R1 = clone_point(P3);
		vec_add_mult(R1,P2,-1.);
		normalize(R1);
		List = overlap(i,s);
		P1 = &glob_mesh.Points[List[0]];
		n1 = get_normal(P3,P2,P1);
		free(List);
		s = glob_mesh.Connections[i][glob_mesh.Sizes[i]-1];
		P2 = &glob_mesh.Points[s];
		List = overlap(i,s);
		P1 = &glob_mesh.Points[List[0]];
		R2 = clone_point(P2);
		vec_add_mult(R2,P3,-1.);
		normalize(R2);
		n2 = get_normal(P3,P2,P1);
		l2 = dist(P3,P2);
		free(List);
		vec_add_mult(&n2,&n1,-1.);
		vec_add_mult(R2,R1,-1.);
		//normalize(R2);
		K = -vec_abs(&n2)*sign(vec_scalar(&n2,&n1));
		free(R1);
		free(R2);
		l = (l1>l2)?l1:l2;
		return K/l;
	}
	else return NAN;
}*/

int* partition2D(mesh2D* Mesh,int* Borders,point2D* Seeds,int n_border,int n_seed){
	int i,j,k,ind,size,new_size;
	int* Pool;
	if (Mesh==NULL) Mesh = &glob_mesh;
	int* Partition = zero_int_list(Mesh->size);
	for (i=1;i<=n_seed;i++){
		ind = get_nearest_index(&(Seeds[i-1]));
		if (in_list(Borders,n_border,ind)<0){
			Pool = (int*)malloc(sizeof(int));
			Pool[0] = ind;
			size = 1;
			while(size>0){
				ind = Pool[size-1];
				new_size = size-1;
				Pool = (int*)realloc(Pool,new_size*sizeof(int));
				Partition[ind] = i;
				for (j=0;j<Mesh->Sizes[ind];j++){
					k = Mesh->Connections[ind][j];
					if (Partition[k]==0 && (in_list(Borders,n_border,k)<0)){
						Partition[k] = i;
						if (in_list(Pool,new_size,k)<0){
							new_size++;
							Pool = (int*)realloc(Pool,new_size*sizeof(int));
							Pool[new_size-1] = k;
						}
					}
				}
				size = new_size;
			}
			free(Pool);
		}
	}
	return Partition;
}

/*double get_mean_node_dist(){
	int i,j,k;
	double sum = 0;
	double num = 0;
	if (glob_mesh.Points!=NULL){
		for(i=0;i<glob_mesh.size;i++){
			for(j=0;j<glob_mesh.Sizes[i];j++){
				k = glob_mesh.Connections[i][j];
				sum += dist(&glob_mesh.Points[i],&glob_mesh.Points[k]);
				num++;
			}
		}
		return (double)sum/num;
	}
	return 0;
}*/

int* get_triangle(point2D* P,int* Last_triangle,double rmax){
	int i,j,k,l;
	int* Res = NULL;
	point2D* A;
	point2D* B;
	point2D* C;
	if (Last_triangle!=NULL){
		A = &glob_mesh.Points[Last_triangle[0]];
		B = &glob_mesh.Points[Last_triangle[1]];
		C = &glob_mesh.Points[Last_triangle[2]];
		if (in_triangle(A,B,C,P)>0){
			Res = (int*)malloc(3*sizeof(int));
			Res[0] = Last_triangle[0];
			Res[1] = Last_triangle[1];
			Res[2] = Last_triangle[2];
			return Res;
		}
	}
	for (i=0;i<glob_mesh.size;i++){
		A = &glob_mesh.Points[i];
		if (rmax>0 && dist(A,P)<rmax){
			k = glob_mesh.Connections[i][0];
			B = &glob_mesh.Points[k];
			l = k;
			for (j=1;j<glob_mesh.Sizes[i];j++){
				k = glob_mesh.Connections[i][j];
				C = &glob_mesh.Points[k];
				if (in_triangle(A,B,C,P)>0){
					Res = (int*)malloc(3*sizeof(int));
					Res[0] = i;
					Res[1] = l;
					Res[2] = k;
					return Res;
				}
				B = C;
				l = k;
			}
			if (is_boundary(i)<0){
				k = glob_mesh.Connections[i][0];
				C = &glob_mesh.Points[k];
				if (in_triangle(A,B,C,P)>0){
					Res = (int*)malloc(3*sizeof(int));
					Res[0] = i;
					Res[1] = l;
					Res[2] = k;
					return Res;
				}
			}
		}
	}
	return Res;
}

void sub_sort2D(point2D* Nodes,int** Subsets,int index,int* sub_size,point2D* Ref,double mean_l){
	int j;						// sortiert die Teilmenge Subsets von Nodes bzgl. Ref in arrays von genau 3 Indices, 
	double a;					// mean_l ist zur Skalierung der Kontrolle, dass die Fläche des Dreiecks nicht null ist
	point2D* A;
	point2D* B;
	double eps = 1E-3;
	int buffer_size = *sub_size;
	int* Ind = (int*)malloc(*sub_size*sizeof(int));
	int* Buffer = (int*)malloc(*sub_size*sizeof(int));
	point2D** Plist = (point2D**)malloc(*sub_size*sizeof(point2D*));
	for (j=0;j<(*sub_size);j++){
		Ind[j] = j;
		Buffer[j] = Subsets[index][j];
		Plist[j] = &(Nodes[Subsets[index][j]]);
	}
	point2D_sort(Ind,Plist,*sub_size,Ref);
	if ((*sub_size)>2){
		*sub_size = 2;
		Subsets[index] = (int*)realloc(Subsets[index],*sub_size*sizeof(int));
		for (j=0;j<(*sub_size);j++) Subsets[index][j] = Buffer[Ind[j]];
		A = &Nodes[Buffer[Ind[0]]];
		B = &Nodes[Buffer[Ind[1]]];
		for (j=(*sub_size);j<buffer_size;j++){
			a = 2.*Area_form2D(A,B,&Nodes[Buffer[Ind[j]]])/(mean_l*mean_l);
			if (fabs(a)>eps){
				*sub_size = 3;
				Subsets[index] = (int*)realloc(Subsets[index],*sub_size*sizeof(int));
				Subsets[index][2] = Buffer[Ind[j]];
				break;
			}
		}
	}
	else{
		for (j=0;j<(*sub_size);j++) Subsets[index][j] = Buffer[Ind[j]];
	}
	Subsets[index] = (int*)realloc(Subsets[index],3*sizeof(int));
	if ((*sub_size)==0) Subsets[index] = NULL;
	else for (j=(*sub_size);j<3;j++) Subsets[index][j] = -1;  // -1 falls nur zwei Punkte ! 
	free(Plist);
	free(Buffer);
	free(Ind);
}

void create_tri_map(point2D* New_nodes,int** Tri_map,int** Inv_tri_map,int size,double r){
	int i;
	/*double d;
	double eps = 1E-3;*/
	int* Last_tri = NULL;
	int* Inv_size = (int*)malloc(glob_mesh.size*sizeof(int));
	for (i=0;i<glob_mesh.size;i++) Inv_size[i] = 0;
	double mean_l = mean_edge_len();
	int counter = 0;
	for (i=0;i<size;i++){
		Tri_map[i] = get_triangle(&New_nodes[i],Last_tri,r*mean_l);
		Last_tri = Tri_map[i];
		if (Last_tri!=NULL){
			Inv_size[Last_tri[0]]++;
			Inv_size[Last_tri[1]]++;
			Inv_size[Last_tri[2]]++;
			Inv_tri_map[Last_tri[0]] = (int*)realloc(Inv_tri_map[Last_tri[0]],Inv_size[Last_tri[0]]*sizeof(int));
			Inv_tri_map[Last_tri[1]] = (int*)realloc(Inv_tri_map[Last_tri[1]],Inv_size[Last_tri[1]]*sizeof(int));
			Inv_tri_map[Last_tri[2]] = (int*)realloc(Inv_tri_map[Last_tri[2]],Inv_size[Last_tri[2]]*sizeof(int));
			Inv_tri_map[Last_tri[0]][Inv_size[Last_tri[0]]-1] = i;
			Inv_tri_map[Last_tri[1]][Inv_size[Last_tri[1]]-1] = i;
			Inv_tri_map[Last_tri[2]][Inv_size[Last_tri[2]]-1] = i;
		}
		else counter++;
	}
	//printf("%d\n",counter);
	for (i=0;i<glob_mesh.size;i++){
		sub_sort2D(New_nodes,Inv_tri_map,i,&(Inv_size[i]),&glob_mesh.Points[i],mean_l);
		/*if (Inv_size[i]<3){
			printf("bad interpolation at index %d\n: %d points\n",i,Inv_size[i]);
		}
		else{
			d = 2.*Area_form2D(&New_nodes[Inv_tri_map[i][0]],&New_nodes[Inv_tri_map[i][1]],&New_nodes[Inv_tri_map[i][2]])/(mean_l*mean_l);
			if (fabs(d)<eps) printf("bad interpolation at index %d\n: eps=%e\n",i,d);
		}*/
	}
	free(Inv_size);
}

void mesh_interpolation(point2D* New_nodes,double* New_values,int size,double* Values,int** Tri_map){
	int i;
	double a,b,c;
	point2D* A;
	point2D* B;
	point2D* C;
	for (i=0;i<size;i++){
		New_values[i] = NAN;
		if (Tri_map[i]!=NULL){
			A = &glob_mesh.Points[Tri_map[i][0]];
			B = &glob_mesh.Points[Tri_map[i][1]];
			C = &glob_mesh.Points[Tri_map[i][2]];
			a = Values[Tri_map[i][0]];
			b = Values[Tri_map[i][1]];
			c = Values[Tri_map[i][2]];
			New_values[i] = linear_interpolation2D(A,B,C,a,b,c,&New_nodes[i]);
		}
	}
}

void inverse_interpolation(point2D* New_nodes,double* New_values,double* Values,int** Inv_tri_map){
	int i;
	double a,b,c;
	point2D* A;
	point2D* B;
	point2D* C;
	for (i=0;i<glob_mesh.size;i++){
		if (Inv_tri_map[i]!=NULL){
			if (Inv_tri_map[i][1]>=0 && Inv_tri_map[i][2]>=0){
				A = &New_nodes[Inv_tri_map[i][0]];
				B = &New_nodes[Inv_tri_map[i][1]];
				C = &New_nodes[Inv_tri_map[i][2]];
				a = New_values[Inv_tri_map[i][0]];
				b = New_values[Inv_tri_map[i][1]];
				c = New_values[Inv_tri_map[i][2]];
				Values[i] = linear_interpolation2D(A,B,C,a,b,c,&glob_mesh.Points[i]);
			}
			else Values[i] = New_values[Inv_tri_map[i][0]];
		}
		else printf("back-interpolation for node %d not possible: error in map\n",i);
	}
}

point2D* create_regular_mesh2D(int n,int m,double rim_size,point2D* Ref){
	int i,j,k;
	point2D min;
	point2D max;
	point2D* Res = (point2D*)malloc(n*m*sizeof(point2D));
	get_mesh_bounds2D(&min,&max,&glob_mesh,rim_size);
	k = 0;
	for (i=0;i<n;i++){
		for (j=0;j<m;j++){
			Res[k].x = min.x+(max.x-min.x)*i/(n-1);
			Res[k].y = min.y+(max.y-min.y)*j/(m-1);
			k++;
		}
	}
	Ref->x = min.x;
	Ref->y = min.y;
	return Res;
}

point2D get_reg_dist(int x_size,int y_size,double rim_size){
	point2D min;
	point2D max;
	point2D res;
	get_mesh_bounds2D(&min,&max,&glob_mesh,rim_size);
	res.x = (double)(max.x-min.x)/(x_size-1);
	res.y = (double)(max.y-min.y)/(y_size-1);
	return res;
}

void refine_mesh(char* filename,double (*area_function)(point2D* P),int quiet){
	int i,j,n,index;
	double a;
	int len = 0;
	char Name[512];
	char Buffer[512];
	index3D tri;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	point2D* P;
	sprintf(Name,"%s.ele",filename);
	if (!quiet) printf("open file %s\n",Name);
	FILE* file_in = fopen(Name,"r");
	
	readline(file_in,Buffer);
	char** line = split(Buffer," ",&len);
	if (len==3){
		n = atoi(line[0]);
		for (j=0;j<len;j++) free(line[j]);free(line);
		sprintf(Name,"%s.area",filename);
		FILE* file_out = fopen(Name,"w");
		fprintf(file_out,"%d\n",n);
		for (i=0;i<n;i++){
			readline(file_in,Buffer);
			line = split(Buffer," ",&len);
			index = atoi(line[0]);
			tri.i = atoi(line[1]);
			tri.j = atoi(line[2]);
			tri.k = atoi(line[3]);
			P1 = &(glob_mesh.Points[tri.i]);
			P2 = &(glob_mesh.Points[tri.j]);
			P3 = &(glob_mesh.Points[tri.k]);
			P = clone_point(P1);
			vec_add_mult(P,P2,1.);
			vec_add_mult(P,P3,1.);
			vec_mult(P,(double)1./3.);
			a = det(P1,P2,P3)*(*area_function)(P);
			fprintf(file_out,"%d\t%e\n",index,a);
			for (j=0;j<len;j++) free(line[j]);free(line);
			free(P);
		}
		fclose(file_out);
		char Command[512];
		sprintf(Command,"triangle -r -q30 -a %s",filename);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
	}
	else{
		printf("error reading file %s/.ele -> abort\n",filename);
		exit(0);
	}
	fclose(file_in);
}

void add_constraint_points(char* FileIn,char* FileOut,point2D* Points,int size){
	int i,j,len,n,n0,success;
	char Line[512];
	char Buffer[512];
	char** Parts;
	
	FILE* In = fopen(FileIn,"r");
	int erri = errno;
	FILE* Out = fopen(FileOut,"w");
	int erro = errno;
		
	if (In!=NULL && Out!=NULL){
		readline(In,Buffer);
		Parts = split(Buffer," ",&len);
		if (len>0){
			n0 = atoi(Parts[0]);
			n = n0+size;
			sprintf(Line,"%d",n);
			free(Parts[0]);
			
			for (j=1;j<len;j++){
				strcat(Line," ");
				strcat(Line,Parts[j]);				
				free(Parts[j]);
			}
			fprintf(Out,"%s\n",Line);
			free(Parts);
			
			for (i=0;i<n0;i++){
				readline(In,Buffer);
				fprintf(Out,"%s\n",Buffer);
			}
			
			for (i=0;i<size;i++){
				fprintf(Out,"%d %.10f %.10f\n",n0+i,Points[i].x,Points[i].y);
			}
			
			do{
				success = readline(In,Buffer);
				if (success) fprintf(Out,"%s\n",Buffer);
			}while(success);
		}
	}
	else{
		if (In==NULL){
			printf("add_constraint_points: could not open file %s -> abort\n",FileIn);
			printf("error number: %d\n",erri);
		}
		if (Out==NULL){
			printf("add_constraint_points: could not open file %s -> abort\n",FileOut);
			printf("error number: %d\n",erro);
		}
		fflush(stdout);	
	}		
	if (In!=NULL) fclose(In);
	if (Out!=NULL) fclose(Out);
}

double get_largest_element_area(mesh2D* Mesh,element_collection* Triangles){
	int i;
	double a;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	index3D ind;
	int N = Triangles->size;
	
	double res = 0;
	for (i=0;i<N;i++){
		ind = Triangles->Elements[i];
		P1 = &(Mesh->Points[ind.i]);
		P2 = &(Mesh->Points[ind.j]);
		P3 = &(Mesh->Points[ind.k]);
		a = fabs(det(P1,P2,P3));
		if (a>res) res = a;
	}
	
	return res;
}

double get_longest_edge(mesh2D* Mesh){
	int i,j,k;
	double d;
	
	double max = 0;
	for (i=0;i<Mesh->size;i++){
		for (j=0;j<Mesh->Sizes[i];j++){
			k = Mesh->Connections[i][j];
			d = dist(&(Mesh->Points[i]),&(Mesh->Points[k]));
			if (d>max) max = d;
		}
	}
	return max;
}

sparse_matrix* get_convolution_matrix(double (*Kernel)(point2D* P1,point2D* P2),double range){
	
	int i,j,k;
	double sum,d,g;
	
	int n = glob_mesh.size;
	double r2 = range*range;
	double* V = set_vector_Ui_0_2D(1);
	int old = get_var_number2D();
	set_var_number2D(1);
	sparse_matrix* Res = sparse_zero(n);
	
	for (i=0;i<n;i++){	
		for (j=0;j<n;j++){
			d = sqrdist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]));
			if (d<r2){
				g = (*Kernel)(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]))*V[j];
				insert_sparse(Res,g,i,j);		
			}
		}
	}
	normalize_row_sum(Res,NULL);
	
	set_var_number2D(old);
	free(V);
	return Res;
}

void Refine_mesh(char* filename,double* Area_data,int size,int quiet){		// Area_data must have length of .ele array
	int i;
	int len = 0;
	char Name[512];
	//char Buffer[512];
	
	sprintf(Name,"%s.area",filename);
	FILE* file_out = fopen(Name,"w");
	fprintf(file_out,"%d\n",size);
	for (i=0;i<size;i++) fprintf(file_out,"%d\t%.15f\n",i,Area_data[i]);
	fclose(file_out);
	
	char Command[512];
	char* Quiet = "-Q";
	char* Nothing = "";
	char* Switch = (quiet) ? Quiet : Nothing;
		
	sprintf(Command,"triangle -r -F -q30 -a %s %s",Switch,filename);
	if (!quiet) printf("execute: %s\n",Command);
	system(Command);

}

double* Field_histogramm(double* X,int resolution){								// first part: area sum
	const double tol = 1e-15;													// tolerance for assuming constant function
	const int DIGITS = 16;														// second part: exponents, base 10 
	const double thres = log10(0.5);
	
	int i,j;																
	double rel,abs,log;
	
	int n = glob_mesh.size;
	double* A = set_vector_Ui_0_2D(1);
	double* Res = zero_vector(2*resolution);
	
	double area = 0;
	double xmin = DBL_MAX;
	double xmax = DBL_MIN;
	for (i=0;i<n;i++){
		abs = fabs(X[i]);
		if (abs>0 && abs<xmin) xmin = abs;
		if (abs>xmax) xmax = abs;
		area += A[i];
	}
	
	if (xmax-xmin>tol){
		double thres_area = 0;
		xmin = log10(xmin);
		xmax = log10(xmax);
		Res[resolution] = (double)xmax-DIGITS-1.;
		xmin = (xmin<Res[resolution]) ? Res[resolution] : xmin;
		for (i=0;i<n;i++){
			abs = fabs(X[i]);					
			if (abs>0 && log10(abs)>=xmin){
				rel = (log10(abs)-xmin)/(xmax-xmin);				
				j = 1+(int)floor((double)rel*(resolution-1));
				j = (j>=resolution) ? resolution-1 : j;
				Res[j] += A[i];
				if (log10(abs)>thres) thres_area += A[i];
			}
			else Res[0] += A[i];
		}
		for (i=1;i<resolution;i++){
			rel = (double)(i-1)/(resolution-1);
			Res[resolution+i] = xmin+(xmax-xmin)*rel;
		}		
		scalar_mult(1./area,&(Res[0]),resolution);
		//printf("area in which x>%f: %e\n",pow(10,thres),thres_area);
	}
	else{
		if (xmax>DBL_MIN){
			log = log10(xmax);
			Res[resolution-1] = 1.;
			Res[resolution] = (double)log-DIGITS-1.;
			for (i=1;i<resolution;i++) Res[resolution+i] = Res[resolution]+(double)(log-Res[resolution])*i/(resolution-1);			
		}
		else{
			Res[0] = 1.;
			for (i=0;i<resolution;i++) Res[resolution+i] = (double)(DBL_MIN_EXP+i);
		}
	}
	
	free(A);
	return Res;
}

void print_histogram(char* Filename,double* Hist,int size){						// length of Hist must be 2*size
	int i;
	
	FILE* file = fopen(Filename,"w");
	if (file!=NULL){
		for (i=0;i<size;i++) fprintf(file,"%f\t%e\n",Hist[size+i],Hist[i]);
		fclose(file);
	}
	else printf("print histogram: could not create file %s -> skip\n",Filename);
}

/*void Refine_mesh(char* filename,double* Area_data,int size,int quiet){
	int i,j,n,index;
	double a,d,d1,d2,d3;
	int len = 0;
	char Name[512];
	char Buffer[512];
	index3D tri;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	sprintf(Name,"%s.ele",filename);
	if (!quiet) printf("open file %s\n",Name);
	FILE* file_in = fopen(Name,"r");
	
	readline(file_in,Buffer);
	char** line = split(Buffer," ",&len);
	if (len==3){
		n = atoi(line[0]);
		for (j=0;j<len;j++) free(line[j]);free(line);
		sprintf(Name,"%s.area",filename);
		FILE* file_out = fopen(Name,"w");
		fprintf(file_out,"%d\n",n);
		for (i=0;i<n;i++){
			readline(file_in,Buffer);
			line = split(Buffer," ",&len);
			index = atoi(line[0]);
			tri.i = atoi(line[1]);
			tri.j = atoi(line[2]);
			tri.k = atoi(line[3]);
			P1 = &(glob_mesh.Points[tri.i]);
			P2 = &(glob_mesh.Points[tri.j]);
			P3 = &(glob_mesh.Points[tri.k]);
			
			d1 = Area_data[tri.i];
			d2 = Area_data[tri.j];
			d3 = Area_data[tri.k];
			d = (d1<d2) ? d1 : d2;
			d = (d3<d) ? d3 : d;
			
			//a = det(P1,P2,P3)*d;											// for relative refinement:
			a = d;															// absolute refinement
			
			fprintf(file_out,"%d\t%.10f\n",index,a);
			for (j=0;j<len;j++) free(line[j]);free(line);
		}
		fclose(file_out);
		char Command[512];
		char* Quiet = "-Q";
		char* Nothing = "";
		char* Switch = (quiet) ? Quiet : Nothing;
		
		sprintf(Command,"triangle -r -F -q30 -a %s %s",Switch,filename);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
	}
	else{
		printf("error reading file %s/.ele -> abort\n",filename);
		exit(0);
	}
	fclose(file_in);
}*/

double* generate_2D_vector(point2D* V,int mesh_size){
	int i;
	double* Res = zero_vector(2*mesh_size);
	for (i=0;i<mesh_size;i++){
		Res[i] = V->x;
		Res[i+mesh_size] = V->y;
	}
	return Res;
}

point2D get_barycentric_2D(point2D* A,point2D* B,point2D* C,point2D* P){		// res.x coordinate A
	point2D res;																// res.y coordinate B
	double det = -B->y*C->x+A->y*(-B->x+C->x)+A->x*(B->y-C->y)+B->x*C->y;		// C = 1-A-B
	res.x = -C->y*P->x+B->y*(-C->x+P->x)+B->x*(C->y-P->y)+C->x*P->y;
	res.y = A->y*(C->x-P->x)+C->y*P->x-C->x*P->y+A->x*(-C->y+P->y);
	vec_mult(&res,1./det);
	return res;
}

void set_linear_coeffcients_2D(point2D* A,point2D* B,point2D* C,double* a,double* b,double* c){		// f(A)=1, f(B)=0, f(C)=0
	double det = C->x*(-A->y+B->y)+B->x*(A->y-C->y)+A->x*(-B->y+C->y);
	if (det==0) printf("set_linear_coeffcients_2D: could not determine coefficients -> skip\n");
	*a = (C->y-B->y)/det;
	*b = (B->x-C->x)/det;
	*c = (C->x*B->y-B->x*C->y)/det;
}

int Inside_triangle(mesh2D* Mesh,element_collection* Triangles,point2D* P,int index,double tol){
	index3D ind = Triangles->Elements[index];
	point2D* A = &(Mesh->Points[ind.i]);
	point2D* B = &(Mesh->Points[ind.j]);
	point2D* C = &(Mesh->Points[ind.k]);
	return in_triangle_2D(A,B,C,P,tol);
}

int inside_triangle(double a,double b, double c){								// a,b,c barycentric coordinates
	const double tol = 1e-8;
	
	if (a<-tol || a>1.+tol) return 0;
	if (b<-tol || b>1.+tol) return 0;
	if (c<-tol || c>1.+tol) return 0;
	return 1;
}

int** get_triangle_list(mesh2D* Mesh,element_collection* Triangles){			// list size in element 0
	int i,j,new_size;															// tri index in 1,2,...,size
	
	int n = Mesh->size;
	int N = Triangles->size;
	
	int** Res = (int**)malloc(n*sizeof(int*));
	for (i=0;i<n;i++) Res[i] = zero_int_list(Mesh->Sizes[i]+1);
		
	for (j=0;j<N;j++){
		i = Triangles->Elements[j].i;
		Res[i][0]++;
		Res[i][Res[i][0]] = j;
		
		i = Triangles->Elements[j].j;
		Res[i][0]++;
		Res[i][Res[i][0]] = j;
		
		i = Triangles->Elements[j].k;
		Res[i][0]++;
		Res[i][Res[i][0]] = j;
	}
	
	for (i=0;i<n;i++){
		new_size = Res[i][0]+1;
		if (new_size<Mesh->Sizes[i]+1) Res[i] = (int*)realloc(Res[i],new_size*sizeof(int));
	}
	
	return Res;
}

int Inside_triangle_set(mesh2D* Mesh,element_collection* Triangles,point2D* P,int* List,double tol){
	int i;
	int size = List[0];
	int res = -1;
	for (i=1;i<=size;i++) if (Inside_triangle(Mesh,Triangles,P,List[i],tol)){
		res = i;
		break;
	}
	return res;
}

void get_xy_extension(mesh2D* Mesh,point2D* Min,point2D* Max){
	const double eps = 1e-6;
	int i;
	double x,y;
	int n = Mesh->size;
	
	Min->x = Mesh->Points[0].x;
	Max->x = Mesh->Points[0].x;
	Min->y = Mesh->Points[0].y;
	Max->y = Mesh->Points[0].y;

	for (i=1;i<n;i++){
		x = Mesh->Points[i].x;
		y = Mesh->Points[i].y;
		if (x<Min->x) Min->x = x;
		if (y<Min->y) Min->y = y;
		if (x>Max->x) Max->x = x;
		if (y>Max->y) Max->y = y;
	}
	Min->x -= eps;
	Min->y -= eps;
	Max->x += eps;
	Max->y += eps;
}

void get_2D_cartasian_partition(mesh2D* Mesh,element_collection* Triangles,point2D* Min,point2D* Max,double d,int**** List,int*** Sizes,int* nx,int* ny){
	int i,j,k,i1,i2,i3,j1,j2,j3;
	double dx,dy;
	point2D* P;
	point2D* P2;
	point2D* P3;
	int n = Triangles->size;
	*nx = round((Max->x-Min->x)/d);
	*ny = round((Max->y-Min->y)/d);
	dx = (double)(Max->x-Min->x)/(*nx);
	dy = (double)(Max->y-Min->y)/(*ny);
	
	*List = (int***)malloc((*nx)*sizeof(int**));
	*Sizes = (int**)malloc((*nx)*sizeof(int*));
	for (i=0;i<(*nx);i++){
		(*List)[i] = (int**)malloc((*ny)*sizeof(int*));
		(*Sizes)[i] = (int*)malloc((*ny)*sizeof(int));
		for (j=0;j<(*ny);j++){
			(*List)[i][j] = NULL;
			(*Sizes)[i][j] = 0;
		}
	}
	//printf("\n");
	for (k=0;k<n;k++){
		index3D ind = Triangles->Elements[k];
		//printf("\rk=%d",k);
		//fflush(stdout);
		P = &(Mesh->Points[ind.i]);
		i1 = (int)floor((P->x-Min->x)/dx);
		j1 = (int)floor((P->y-Min->y)/dy);
		(*Sizes)[i1][j1]++;
		(*List)[i1][j1] = (int*)realloc((*List)[i1][j1],(*Sizes)[i1][j1]*sizeof(int));
		(*List)[i1][j1][(*Sizes)[i1][j1]-1] = k;
		
		P = &(Mesh->Points[ind.j]);
		i2 = (int)floor((P->x-Min->x)/dx);
		j2 = (int)floor((P->y-Min->y)/dy);
		if (i2!=i1 || j2!=j1){
			(*Sizes)[i2][j2]++;
			(*List)[i2][j2] = (int*)realloc((*List)[i2][j2],(*Sizes)[i2][j2]*sizeof(int));
			(*List)[i2][j2][(*Sizes)[i2][j2]-1] = k;
		}
		
		P = &(Mesh->Points[ind.k]);
		i3 = (int)floor((P->x-Min->x)/dx);
		j3 = (int)floor((P->y-Min->y)/dy);
		if ((i3!=i1 || j3!=j1) && (i3!=i2 || j3!=j)){
			(*Sizes)[i3][j3]++;
			(*List)[i3][j3] = (int*)realloc((*List)[i3][j3],(*Sizes)[i3][j3]*sizeof(int));
			(*List)[i3][j3][(*Sizes)[i3][j3]-1] = k;
		}
	} 
}

int compare_point_dist(void* X1,void* X2){
	static point2D* P = NULL;
	static mesh2D* Mesh = NULL;
	
	if (X1!=NULL && X2!=NULL){
		index3D* Ind1 = (index3D*)X1;											// compare
		point2D p1 = init_point2D(0,0);
		vec_add_mult(&p1,&(Mesh->Points[Ind1->i]),(double)1/3);
		vec_add_mult(&p1,&(Mesh->Points[Ind1->j]),(double)1/3);
		vec_add_mult(&p1,&(Mesh->Points[Ind1->k]),(double)1/3);
		double d1 = dist(&p1,P);
		
		index3D* Ind2 = (index3D*)X2;
		point2D p2 = init_point2D(0,0);
		vec_add_mult(&p2,&(Mesh->Points[Ind2->i]),(double)1/3);
		vec_add_mult(&p2,&(Mesh->Points[Ind2->j]),(double)1/3);
		vec_add_mult(&p2,&(Mesh->Points[Ind2->k]),(double)1/3);
		double d2 = dist(&p2,P);
		
		if (d1<=d2) return -1; else return 1;
	}
	else{
		if (X1==NULL) P = (point2D*)X2;											// init 
		if (X2==NULL) Mesh = (mesh2D*)X1;
		return 0;
	}
}

void search_whole_domain(point2D* P,mesh2D* Mesh,element_collection* Triangles,sparse_matrix* Interpolation,int row_index){
	int i,i0,j0,k0;
	double a,b,c,a0,b0,c0;
	point2D bar;
	index3D* Ind;
	
	int n = Triangles->size;
	int* Local_list = get_ID_index_map(n);
	void** Element_list = (void**)malloc(n*sizeof(void*));
	for (i=0;i<n;i++) Element_list[i] = &(Triangles->Elements[i]);
	
	compare_point_dist(NULL,P);
	compare_point_dist(Mesh,NULL);
	divide_and_conquer_sort(Local_list,Element_list,n,&compare_point_dist);	
	
	for (i=0;i<n;i++){
		Ind = &(Triangles->Elements[Local_list[i]]);
		bar = get_barycentric_2D(&(Mesh->Points[Ind->i]),&(Mesh->Points[Ind->j]),&(Mesh->Points[Ind->k]),P);
		a = bar.x;
		b = bar.y;
		c = 1.-a-b;
		if (i==0){
			a0 = a;
			b0 = b;
			c0 = c;
			i0 = Ind->i;
			j0 = Ind->j;
			k0 = Ind->k;
		}
		if (inside_triangle(a,b,c)){
			insert_sparse(Interpolation,a,row_index,Ind->i);
			insert_sparse(Interpolation,b,row_index,Ind->j);
			insert_sparse(Interpolation,c,row_index,Ind->k);
			break;
		}				
	}
	if (i==n){
		printf("Warning: no triangle found for point (%f,%f)\n",P->x,P->y);
		printf("extrapolate with nearest element: (%f,%f,%f)\n\n",a0,b0,c0);
		insert_sparse(Interpolation,a0,row_index,i0);
		insert_sparse(Interpolation,b0,row_index,j0);
		insert_sparse(Interpolation,c0,row_index,k0);
	}	
	
	free(Local_list);
	free(Element_list);
}

void cartesian_partition(mesh2D* Mesh,element_collection* Triangles,point2D ref,double dx,double dy,int*** Lists,int** Sizes){
	int k,ix,iy,jx,jy,kx,ky,lx,ly;
	double x,y;
	index3D ind;
	point2D* Pi;
	point2D* Pj;
	point2D* Pk;
	
	for (k=0;k<Triangles->size;k++){
		ind = Triangles->Elements[k];
		Pi = &(Mesh->Points[ind.i]);
		Pj = &(Mesh->Points[ind.j]);
		Pk = &(Mesh->Points[ind.k]);
		
		ix = (int)floor((Pi->x-ref.x)/dx);
		iy = (int)floor((Pi->y-ref.y)/dy);
		
		jx = (int)floor((Pj->x-ref.x)/dx);
		jy = (int)floor((Pj->y-ref.y)/dy);
		
		kx = (int)floor((Pk->x-ref.x)/dx);
		ky = (int)floor((Pk->y-ref.y)/dy);
		
		if (((ix-jx)*(ix-jx)+(iy-jy)*(iy-jy)==0) && ((ix-kx)*(ix-kx)+(iy-ky)*(iy-ky)==0)){
			Sizes[ix][iy]++;
			Lists[ix][iy] = (int*)realloc(Lists[ix][iy],Sizes[ix][iy]*sizeof(int));
			Lists[ix][iy][Sizes[ix][iy]-1] = k;
		}
		else{
			if ((ix-jx)*(ix-jx)+(iy-jy)*(iy-jy)==0){
				Sizes[ix][iy]++;
				Lists[ix][iy] = (int*)realloc(Lists[ix][iy],Sizes[ix][iy]*sizeof(int));
				Lists[ix][iy][Sizes[ix][iy]-1] = k;
				Sizes[kx][ky]++;
				Lists[kx][ky] = (int*)realloc(Lists[kx][ky],Sizes[kx][ky]*sizeof(int));
				Lists[kx][ky][Sizes[kx][ky]-1] = k;
				if ((kx-ix)*(kx-ix)+(ky-iy)*(ky-iy)>=2){
					Sizes[ix][ky]++;
					Lists[ix][ky] = (int*)realloc(Lists[ix][ky],Sizes[ix][ky]*sizeof(int));
					Lists[ix][ky][Sizes[ix][ky]-1] = k;
					Sizes[kx][iy]++;
					Lists[kx][iy] = (int*)realloc(Lists[kx][iy],Sizes[kx][iy]*sizeof(int));
					Lists[kx][iy][Sizes[kx][iy]-1] = k;					
				}				
				goto FINALIZE;
			}
			if (((ix-kx)*(ix-kx)+(iy-ky)*(iy-ky)==0) || ((jx-kx)*(jx-kx)+(jy-ky)*(jy-ky)==0)){
				Sizes[ix][iy]++;
				Lists[ix][iy] = (int*)realloc(Lists[ix][iy],Sizes[ix][iy]*sizeof(int));
				Lists[ix][iy][Sizes[ix][iy]-1] = k;
				Sizes[jx][jy]++;
				Lists[jx][jy] = (int*)realloc(Lists[jx][jy],Sizes[jx][jy]*sizeof(int));
				Lists[jx][jy][Sizes[jx][jy]-1] = k;
				if ((jx-ix)*(jx-ix)+(jy-iy)*(jy-iy)>=2){
					Sizes[ix][jy]++;
					Lists[ix][jy] = (int*)realloc(Lists[ix][jy],Sizes[ix][jy]*sizeof(int));
					Lists[ix][jy][Sizes[ix][jy]-1] = k;
					Sizes[jx][iy]++;
					Lists[jx][iy] = (int*)realloc(Lists[jx][iy],Sizes[jx][iy]*sizeof(int));
					Lists[jx][iy][Sizes[jx][iy]-1] = k;					
				}
				goto FINALIZE;				
			}
			
			Sizes[ix][iy]++;
			Lists[ix][iy] = (int*)realloc(Lists[ix][iy],Sizes[ix][iy]*sizeof(int));
			Lists[ix][iy][Sizes[ix][iy]-1] = k;
			Sizes[jx][jy]++;
			Lists[jx][jy] = (int*)realloc(Lists[jx][jy],Sizes[jx][jy]*sizeof(int));
			Lists[jx][jy][Sizes[jx][jy]-1] = k;
			Sizes[kx][ky]++;
			Lists[kx][ky] = (int*)realloc(Lists[kx][ky],Sizes[kx][ky]*sizeof(int));
			Lists[kx][ky][Sizes[kx][ky]-1] = k;
			
			x = (double)(ix+jx+kx)/3;
			y = (double)(iy+jy+ky)/3;
			lx = (int)floor(x);
			ly = (int)floor(y);
			x -= (double)lx;
			y -= (double)ly;
			if (x<0.5) lx++;
			if (y<0.5) ly++;
			
			Sizes[lx][ly]++;
			Lists[lx][ly] = (int*)realloc(Lists[lx][ly],Sizes[lx][ly]*sizeof(int));
			Lists[lx][ly][Sizes[lx][ly]-1] = k;
			
			FINALIZE:
			x = 0;																		// dummy operation
		}
	}
	
}

void find_cartesian_block(mesh2D* Mesh,element_collection* Triangles,point2D ref,double dx,double dy,index2D** Block_list,int* Block_num){
	int k,ix,iy,jx,jy,kx,ky,lx,ly;
	double x,y;
	index3D ind;
	point2D* Pi;
	point2D* Pj;
	point2D* Pk;
	
	for (k=0;k<Triangles->size;k++){
		ind = Triangles->Elements[k];
		Pi = &(Mesh->Points[ind.i]);
		Pj = &(Mesh->Points[ind.j]);
		Pk = &(Mesh->Points[ind.k]);
		
		ix = (int)floor((Pi->x-ref.x)/dx);
		iy = (int)floor((Pi->y-ref.y)/dy);
		
		jx = (int)floor((Pj->x-ref.x)/dx);
		jy = (int)floor((Pj->y-ref.y)/dy);
		
		kx = (int)floor((Pk->x-ref.x)/dx);
		ky = (int)floor((Pk->y-ref.y)/dy);
		
		if (((ix-jx)*(ix-jx)+(iy-jy)*(iy-jy)==0) && ((ix-kx)*(ix-kx)+(iy-ky)*(iy-ky)==0)){
			Block_num[k] = 1;
			Block_list[k] = (index2D*)malloc(sizeof(index2D));
			Block_list[k][0].i = ix;
			Block_list[k][0].j = iy;
		}
		else{
			if ((ix-jx)*(ix-jx)+(iy-jy)*(iy-jy)==0){
				
				if ((kx-ix)*(kx-ix)+(ky-iy)*(ky-iy)>=2) Block_num[k] = 4; else Block_num[k] = 2;
				Block_list[k] = (index2D*)malloc(Block_num[k]*sizeof(index2D));
				Block_list[k][0].i = ix;
				Block_list[k][0].j = iy;
				Block_list[k][1].i = kx;
				Block_list[k][1].j = ky;
				if (Block_num[k]==4){
					Block_list[k][2].i = ix;
					Block_list[k][2].j = ky;
					Block_list[k][3].i = kx;
					Block_list[k][3].j = iy;
				}
				goto FINALIZE;
			}
			if (((ix-kx)*(ix-kx)+(iy-ky)*(iy-ky)==0) || ((jx-kx)*(jx-kx)+(jy-ky)*(jy-ky)==0)){								
				if ((jx-ix)*(jx-ix)+(jy-iy)*(jy-iy)>=2) Block_num[k] = 4; else Block_num[k] = 2;
				Block_list[k] = (index2D*)malloc(Block_num[k]*sizeof(index2D));
				Block_list[k][0].i = ix;
				Block_list[k][0].j = iy;
				Block_list[k][1].i = jx;
				Block_list[k][1].j = jy;
				if (Block_num[k]==4){
					Block_list[k][2].i = ix;
					Block_list[k][2].j = jy;
					Block_list[k][3].i = jx;
					Block_list[k][3].j = iy;
				}
				goto FINALIZE;
			}
			
			Block_num[k] = 4;
			Block_list[k] = (index2D*)malloc(4*sizeof(index2D));		
			Block_list[k][0].i = ix;
			Block_list[k][0].j = iy;
			Block_list[k][1].i = jx;
			Block_list[k][1].j = jy;
			Block_list[k][2].i = kx;
			Block_list[k][2].j = ky;											
			
			x = (double)(ix+jx+kx)/3;
			y = (double)(iy+jy+ky)/3;
			lx = (int)floor(x);
			ly = (int)floor(y);
			x -= (double)lx;
			y -= (double)ly;
			if (x<0.5) lx++;
			if (y<0.5) ly++;
			Block_list[k][3].i = lx;
			Block_list[k][3].j = ly;
			
			FINALIZE:
			x = 0;																		// dummy operation
		}
	}	
}

double line_integral_product(point2D* P1,point2D* P2,double a1,double b1,double c1,double a2,double b2,double c2){
	const double tol = 1e-10;
	if (fabs(P2->y-P1->y)<tol) return 0; 
	else{
		double res;
		double y1 = P1->y;
		double y2 = P2->y;
		double m = (P2->x-P1->x)/(y2-y1);
		double n = P1->x-m*y1;
		
		double p = m*(3.*b1*(2.*b2+a2*m)+a1*m*(3.*b2+2.*a2*m))/6.;
		double q = m*(2.*b2*c1+2.*b1*c2+a2*c1*m+a1*c2*m)/2.+(b1+a1*m)*(b2+a2*m)*n;
		double r = c2*(b1+a1*m)*n+(a1*b2+a2*(b1+2.*a1*m))*n*n/2.+c1*(c2*m+(b2+a2*m)*n);
		double s = n*(3.*c1*(2.*c2+a2*n)+a1*n*(3.*c2+2.*a2*n))/6.;
		
		res = s*(y2-y1);
		y1 *= P1->y;
		y2 *= P2->y;
		res += r*(y2-y1)/2.;
		y1 *= P1->y;
		y2 *= P2->y;
		res += q*(y2-y1)/3.;
		y1 *= P1->y;
		y2 *= P2->y;
		res += p*(y2-y1)/4.;
		return res;
	}
}

void feed_line_int(){
	double a1 = 1.;
	double b1 = -1.;
	double c1 = 1.;
	double a2 = 1.;
	double b2 = -2.;
	double c2 = -1.;
	point2D p1 = init_point2D(0,0);
	point2D p2 = init_point2D(0.5,0);
	point2D p3 = init_point2D(1.,0.3333333333);
	point2D p4 = init_point2D(1.,1.);
	point2D p5 = init_point2D(0,0.33333333333);
	double q = line_integral_product(&p1,&p2,a1,b1,c1,a2,b2,c2);
	q += line_integral_product(&p2,&p3,a1,b1,c1,a2,b2,c2);
	q += line_integral_product(&p3,&p4,a1,b1,c1,a2,b2,c2);
	q += line_integral_product(&p4,&p5,a1,b1,c1,a2,b2,c2);
	q += line_integral_product(&p5,&p1,a1,b1,c1,a2,b2,c2);
	printf("%e\n",q);
	exit(0);
}

/*double get_linear_poly_value(point2D* P,double* Coeffs){
	return Coeffs[0]*P->x+Coeffs[1]*P->y+Coeffs[2];
}*/

double* transform_linear_coeffcients_2D(double* Matrix,double* Offset,double a,double b,double c){		// A->(1,0),B->(0,1),C->(0,0)
	double* Res = zero_vector(3);
	Res[0] = a*Matrix[0]+b*Matrix[2];
	Res[1] = a*Matrix[1]+b*Matrix[3];
	Res[2] = c+a*Offset[0]+b*Offset[1];
	return Res;
}

double unity_triangle_integral(double* Coeff1,double* Coeff2){
	return (Coeff1[0]*(2.*Coeff2[0]+Coeff2[1]+4.*Coeff2[2])+Coeff1[1]*(Coeff2[0]+2*Coeff2[1]+4.*Coeff2[2])+4.*Coeff1[2]*(Coeff2[0]+Coeff2[1]+3.*Coeff2[2]))/24.;
}

double tri_int(mesh2D* Mesh,element_collection* Tris,int index){
	index3D ind = Tris->Elements[index];
	point2D* A = &(Mesh->Points[ind.i]);
	point2D* B = &(Mesh->Points[ind.j]);
	point2D* C = &(Mesh->Points[ind.k]);
	return fabs(det(A,B,C))/6.;
}

double* get_overlap_integral_matrix(point2D* A1,point2D* B1,point2D* C1,point2D* A2,point2D* B2,point2D* C2,point2D** Poly,int poly_size){
	const int SIZE = 3;
	int i,j,k;
	double detF;
	point2D* P1;
	point2D* P2;
	double* Tra1;
	double* Tra2;
	
	double* Affine_matrix = NULL;
	double* Affine_offset = NULL;
	
	// get linear coefficients
	double* Coeffs = zero_vector(2*SIZE*SIZE);
	
	set_linear_coeffcients_2D(A1,B1,C1,&(Coeffs[0]),&(Coeffs[1]),&(Coeffs[2]));
	set_linear_coeffcients_2D(B1,C1,A1,&(Coeffs[3]),&(Coeffs[4]),&(Coeffs[5]));
	set_linear_coeffcients_2D(C1,A1,B1,&(Coeffs[6]),&(Coeffs[7]),&(Coeffs[8]));
	
	set_linear_coeffcients_2D(A2,B2,C2,&(Coeffs[9]),&(Coeffs[10]),&(Coeffs[11]));
	set_linear_coeffcients_2D(B2,C2,A2,&(Coeffs[12]),&(Coeffs[13]),&(Coeffs[14]));
	set_linear_coeffcients_2D(C2,A2,B2,&(Coeffs[15]),&(Coeffs[16]),&(Coeffs[17]));
	
	// get orientation of polygon
	double orientation = 0;
	for (k=0;k<poly_size;k++){
		P1 = Poly[k];
		P2 = (k<poly_size-1) ? Poly[k+1] : Poly[0];
		orientation += (P2->x-P1->x)*(P2->y+P1->y);
	}
	
	// compute integral
	double* Res = zero_vector(SIZE*SIZE);
	if (poly_size==3){
		detF = fabs(det(Poly[0],Poly[1],Poly[2]));
		get_affine_trafo_2D(Poly[0],Poly[1],Poly[2],&Affine_matrix,&Affine_offset);
		for (i=0;i<SIZE;i++){
			for (j=0;j<SIZE;j++){
				Tra1 = transform_linear_coeffcients_2D(Affine_matrix,Affine_offset,Coeffs[SIZE*i],Coeffs[SIZE*i+1],Coeffs[SIZE*i+2]);
				Tra2 = transform_linear_coeffcients_2D(Affine_matrix,Affine_offset,Coeffs[SIZE*SIZE+SIZE*j],Coeffs[SIZE*SIZE+SIZE*j+1],Coeffs[SIZE*SIZE+SIZE*j+2]);
				Res[i*SIZE+j] = unity_triangle_integral(Tra1,Tra2)*detF;
				free(Tra1);
				free(Tra2);
			}
		}
		free(Affine_matrix);
		free(Affine_offset);
	}
	else{
		point2D center = init_point2D(0,0);
		for (k=0;k<poly_size;k++) vec_add_mult(&center,Poly[k],1.);
		vec_mult(&center,(double)1/poly_size);
		
		for (k=0;k<poly_size;k++){
			P1 = Poly[k];
			P2 = (k<poly_size-1) ? Poly[k+1] : Poly[0];
			
			detF = fabs(det(P1,P2,&center));
			get_affine_trafo_2D(P1,P2,&center,&Affine_matrix,&Affine_offset);
			
			for (i=0;i<SIZE;i++){
				for (j=0;j<SIZE;j++){
					Tra1 = transform_linear_coeffcients_2D(Affine_matrix,Affine_offset,Coeffs[SIZE*i],Coeffs[SIZE*i+1],Coeffs[SIZE*i+2]);
					Tra2 = transform_linear_coeffcients_2D(Affine_matrix,Affine_offset,Coeffs[SIZE*SIZE+SIZE*j],Coeffs[SIZE*SIZE+SIZE*j+1],Coeffs[SIZE*SIZE+SIZE*j+2]);
					Res[i*SIZE+j] += unity_triangle_integral(Tra1,Tra2)*detF;
					free(Tra1);
					free(Tra2);
				}
			}
			free(Affine_matrix);
			free(Affine_offset);
		}
	}
	
	/*double* Res = zero_vector(SIZE*SIZE);
	for (i=0;i<SIZE;i++){		
		for (j=0;j<SIZE;j++){		
			for (k=0;k<poly_size;k++){
				P1 = Poly[k];
				P2 = (k<poly_size-1) ? Poly[k+1] : Poly[0];
				Res[i*SIZE+j] += line_integral_product(P1,P2,Coeffs[i*SIZE],Coeffs[i*SIZE+1],Coeffs[i*SIZE+2],
				 Coeffs[SIZE*SIZE+j*SIZE],Coeffs[SIZE*SIZE+j*SIZE+1],Coeffs[SIZE*SIZE+j*SIZE+2]);
			}
			Res[i*SIZE+j] *= -sign(orientation);
			//Res[i*SIZE+j] = fabs(Res[i*SIZE+j]);
		}
	}*/
		
	free(Coeffs);
	return Res;
}

double max_triangle_size(mesh2D* Mesh,element_collection* Triangles){
	const double factor = 1.1;
	
	int i;
	index3D ind;
	point2D A;
	point2D B;
	point2D C;
	
	int N = Triangles->size;
	double max = 0;
	for (i=0;i<N;i++){
		ind = Triangles->Elements[i];
		A = Mesh->Points[ind.i];
		B = Mesh->Points[ind.j];
		C = Mesh->Points[ind.k];
		
		if (fabs(A.x-B.x)>max) max = fabs(A.x-B.x);
		if (fabs(B.x-C.x)>max) max = fabs(B.x-C.x);
		if (fabs(C.x-A.x)>max) max = fabs(C.x-A.x);
		
		if (fabs(A.y-B.y)>max) max = fabs(A.y-B.y);
		if (fabs(B.y-C.y)>max) max = fabs(B.y-C.y);
		if (fabs(C.y-A.y)>max) max = fabs(C.y-A.y);
	}
		
	return factor*max;
}

void print_partition(int nx,int ny,int** Sizes){
	int i,j;
	
	printf("cartesian partition:\n");
	printf("blocks: #x = %d, #y=%d\n",nx,ny);
	
	for (i=0;i<nx;i++){
		for (j=0;j<ny;j++){
			printf("block (%d,%d): %d elements\n",i,j,Sizes[i][j]);
		}
	}
	printf("\n");
	fflush(stdout);
}

sparse_matrix* FEM2D_interpolation_matrix(mesh2D* Dest,mesh2D* Source,element_collection* Dest_elements,
 element_collection* Source_elements,double d,double tol){
	 
	 const double conservation_tolerance = 0.03; 								// 3%
	
	//FILE* Out = fopen("/Home/damage/radszuwe/Daten/polys","w");
	int i,j,k,l,s,ix,iy,poly_size;
	index3D source_ind,dest_ind;
	double* R;
	point2D* A1;
	point2D* A2;
	point2D* A3;
	point2D* B1;
	point2D* B2;
	point2D* B3;
	point2D** Poly;
	point2D min = init_point2D(0,0);
	point2D max = init_point2D(0,0);
	get_xy_extension(Source,&min,&max);
	int nx = round((max.x-min.x)/d);
	int ny = round((max.y-min.y)/d);
	double dx = (double)(max.x-min.x)/nx;
	double dy = (double)(max.y-min.y)/ny;
	int m = Source_elements->size;
	int n = Source->size;
	int M = Dest_elements->size;
	int N = Dest->size;
	
	// partition all triangles in source mesh in a cartesian grid
	int** Sizes_Source = (int**)malloc(nx*sizeof(int*));
	int*** Lists_Source = (int***)malloc(nx*sizeof(int**));
	for (i=0;i<nx;i++){
		Lists_Source[i] = (int**)malloc(ny*sizeof(int*));
		for(j=0;j<ny;j++) Lists_Source[i][j] = NULL;
		Sizes_Source[i] = zero_int_list(ny);
	}
	cartesian_partition(Source,Source_elements,min,dx,dy,Lists_Source,Sizes_Source);
	
	// get all partition blocks that may intersect with dest triangles 
	index2D** Block_list = (index2D**)malloc(M*sizeof(index2D*));
	for (i=0;i<M;i++) Block_list[i] = NULL;
	int* Block_num = zero_int_list(M);
	find_cartesian_block(Dest,Dest_elements,min,dx,dy,Block_list,Block_num);
	
	sparse_matrix* Res = sparse_zero(N);
	int* Finished = zero_int_list(m);
	for (k=0;k<M;k++){
		dest_ind = Dest_elements->Elements[k];
		B1 = &(Dest->Points[dest_ind.i]);
		B2 = &(Dest->Points[dest_ind.j]);
		B3 = &(Dest->Points[dest_ind.k]);
		for (i=0;i<Block_num[k];i++){
			ix = Block_list[k][i].i;
			iy = Block_list[k][i].j;
			for (j=0;j<Sizes_Source[ix][iy];j++){
				l = Lists_Source[ix][iy][j];
				if (!Finished[l]){
					source_ind = Source_elements->Elements[l];
					A1 = &(Source->Points[source_ind.i]);
					A2 = &(Source->Points[source_ind.j]);
					A3 = &(Source->Points[source_ind.k]);
					Poly = NULL;
					poly_size = 0;				
					if (chance_to_overlap(A1,A2,A3,B1,B2,B3)) triangle_intersection_2D(A1,A2,A3,B1,B2,B3,&Poly,&poly_size,tol);						
					if (Poly!=NULL){
						R = get_overlap_integral_matrix(B1,B2,B3,A1,A2,A3,Poly,poly_size);
						
						insert_sparse(Res,R[0],dest_ind.i,source_ind.i);
						insert_sparse(Res,R[1],dest_ind.i,source_ind.j);
						insert_sparse(Res,R[2],dest_ind.i,source_ind.k);
						insert_sparse(Res,R[3],dest_ind.j,source_ind.i);
						insert_sparse(Res,R[4],dest_ind.j,source_ind.j);
						insert_sparse(Res,R[5],dest_ind.j,source_ind.k);
						insert_sparse(Res,R[6],dest_ind.k,source_ind.i);
						insert_sparse(Res,R[7],dest_ind.k,source_ind.j);
						insert_sparse(Res,R[8],dest_ind.k,source_ind.k);
												
						/*if (poly_size>6){
							counter++;							
							for (s=0;s<poly_size;s++) fprintf(Out,"%f\t",Poly[s]->x);
							fprintf(Out,"\n");
							for (s=0;s<poly_size;s++) fprintf(Out,"%f\t",Poly[s]->y);
							fprintf(Out,"\n");
						}*/						
						
						for (s=0;s<poly_size;s++) free(Poly[s]);
						free(Poly);
						free(R);
					}
					Finished[l] = 1;
				}
			}
		}
		for (i=0;i<Block_num[k];i++){
			ix = Block_list[k][i].i;
			iy = Block_list[k][i].j;
			for (j=0;j<Sizes_Source[ix][iy];j++) Finished[Lists_Source[ix][iy][j]] = 0;
		}
	}
	
	remove_zeros(Res);
	double* V_old = Get_vector_Ui_0_2D(Source);
	double* V_new = Get_vector_Ui_0_2D(Dest);	
	normalize_row_sum(Res,V_new);												// normalize row sums to ensure stability
	
	double* W_old = get_col_sums(Res,n);
	vector_add(W_old,V_old,-1.,n);
	vector_pseudo_div(W_old,V_old,n);
	double r = euklid_norm(W_old,n);
	if (r>conservation_tolerance) printf("\nWarning: inaccurate interpolation: colomn conservation difference: %e\n",r);
	
	free(V_old);
	free(V_new);
	free(W_old);
	
	//fclose(Out);
	
	// clean
	for (i=0;i<nx;i++){		
		for (j=0;j<ny;j++) if (Lists_Source[i][j]!=NULL) free(Lists_Source[i][j]);
		free(Lists_Source[i]);
		free(Sizes_Source[i]);
	}
	free(Lists_Source);
	free(Sizes_Source);
	for (i=0;i<M;i++) if (Block_list[i]!=NULL) free(Block_list[i]);
	free(Block_list);
	free(Block_num);
	free(Finished);
		
	return Res;
}

double* tri_to_vert(mesh2D* Mesh,element_collection* Triangles,double* Data){
	int i;
	index3D ind;
	double* Res = zero_vector(Mesh->size);
	for (i=0;i<Triangles->size;i++){
		ind = Triangles->Elements[i];
		Res[ind.i] = Data[i];
		Res[ind.j] = Data[i];
		Res[ind.k] = Data[i];
	}
	return Res;
}

/*double* FEM_get_minimum_interpolation(double* X,mesh2D* Dest,element_collection* Dest_elements,mesh2D* Source,double d){
	const double tol = 1e-15;
	
	int i,j,k,l,s;
	double x1,x2,x3;
	point2D a,b,c,p,q;
	index3D ind;															// only valid for positive values
	
	int n = Source->size;
	int N = Dest->size;
	int M = Dest_elements->size;
	point2D min = init_point2D(0,0);
	point2D max = init_point2D(0,0);
	get_xy_extension(Dest,&min,&max);
	int nx = (int)floor((max.x-min.x)/d);
	int ny = (int)floor((max.y-min.y)/d);
	double dx = (double)(max.x-min.x)/nx;
	double dy = (double)(max.y-min.y)/ny;
	double* Res = generate_vector(M,1./tol);
	
	// case if meshes are identical
	if (Dest==Source){
		for (k=0;k<M;k++){
			ind = Dest_elements->Elements[k];
			if (X[ind.i]<X[ind.j]) Res[k] = (X[ind.i]<X[ind.k]) ? X[ind.i] : X[ind.k];
			else Res[k] = (X[ind.j]<X[ind.k]) ? X[ind.j] : X[ind.k];
		}
		return Res;
	}
	
	// if not 
	int** Size_list = (int**)malloc(nx*sizeof(int*));
	int*** Index_list = (int***)malloc(nx*sizeof(int**));
	for (i=0;i<nx;i++){
		Size_list[i] = zero_int_list(ny);
		Index_list[i] = (int**)malloc(ny*sizeof(int*));
		for (j=0;j<ny;j++) Index_list[i][j] = NULL;
	}
	
	// generate cartesian partition
	for (k=0;k<n;k++){
		p = Source->Points[k];
		i = (int)floor((p.x-min.x)/dx);
		j = (int)floor((p.y-min.y)/dy);
		if (i<0 || i>=nx || j<0 || j>=ny){
			printf("get_ninimum_interpolation: index out of range -> abort\n");
			exit(0);
		}
		Size_list[i][j]++;
		Index_list[i][j] = (int*)realloc(Index_list[i][j],Size_list[i][j]*sizeof(int));
		Index_list[i][j][Size_list[i][j]-1] = k;
	}
	
	// find points in triangles
	for (k=0;k<M;k++){
		ind = Dest_elements->Elements[k];
		a = Dest->Points[ind.i];
		b = Dest->Points[ind.j];
		c = Dest->Points[ind.k];
		
		i = (int)floor((a.x-min.x)/dx);
		j = (int)floor((a.y-min.y)/dy);
		for (l=0;l<Size_list[i][j];l++){
			s = Index_list[i][j][l];
			p = Source->Points[s];
			if (in_triangle_2D(&a,&b,&c,&p,tol)){
				if (fabs(X[s])<Res[k]) Res[k] = X[s];
			}
		}
			
		i = (int)floor((b.x-min.x)/dx);
		j = (int)floor((b.y-min.y)/dy);
		for (l=0;l<Size_list[i][j];l++){
			s = Index_list[i][j][l];
			p = Source->Points[s];
			if (in_triangle_2D(&a,&b,&c,&p,tol)){
				if (fabs(X[s])<Res[k]) Res[k] = X[s];
			}
		}
		
		i = (int)floor((c.x-min.x)/dx);
		j = (int)floor((c.y-min.y)/dy);
		for (l=0;l<Size_list[i][j];l++){
			s = Index_list[i][j][l];
			p = Source->Points[s];
			if (in_triangle_2D(&a,&b,&c,&p,tol)){
				if (fabs(X[s])<Res[k]) Res[k] = X[s];
			}
		}
	}
	
	for (i=0;i<M;i++) if (Res[i]>0.1/tol) Res[i] = NAN;
	
	// clean
	for (i=0;i<nx;i++){
		for (j=0;j<ny;j++) free(Index_list[i][j]);
		free(Index_list[i]);
		free(Size_list[i]);
	}
	free(Index_list);
	free(Size_list);
	
	return Res;
}*/

void add_iso_segment(isoline* Iso,index3D ind,double a,double b,double c){
	(Iso->size)++;
	Iso->Tri_indices = (index3D*)realloc(Iso->Tri_indices,(Iso->size)*sizeof(index3D));
	Iso->Coordinates = (point2D*)realloc(Iso->Coordinates,(Iso->size)*sizeof(point2D));
	
	Iso->Tri_indices[(Iso->size)-1] = ind;
	point2D p = {.x = a,.y = b};
	Iso->Coordinates[(Iso->size)-1] = p;
}

isoline* get_isoline(mesh2D* Mesh,double* X,int start_index,double tol,int max_len){
	int i,j,na1,na2,center,new_index,counter,start,bound_encountered,first_size;
	int* list;
	index3D tri;
	index3D start_tri[2];
	double a,b,c;
	point2D G;
	double* New;
		
	isoline* Res = (isoline*)malloc(sizeof(isoline));
	Res->size = 0;
	Res->Tri_indices = NULL;
	Res->Coordinates = NULL;
	
	counter = 0;	
	for (i=0;i<Mesh->Sizes[start_index];i++){		
		na1 = Mesh->Connections[start_index][i];
		if(i==Mesh->Sizes[start_index]-1){		
			if (is_boundary(start_index)>=0) break; else na2 = Mesh->Connections[start_index][0];		
		}
		else na2 = Mesh->Connections[start_index][i+1];
		 
		if ((X[na1]<=X[start_index] && X[na2]>X[start_index]) || (X[na1]>=X[start_index] && X[na2]<X[start_index])){			
			start_tri[counter].i = start_index;
			start_tri[counter].j = na1;
			start_tri[counter].k = na2;		
			counter++;
			if (counter==2) break;
		} 
	}
	
	/*printf("%d tris found\n",counter);
	printf("tri1: (%d,%d,%d)\n",start_tri[0].i,start_tri[0].j,start_tri[0].k);
	printf("tri2: (%d,%d,%d)\n",start_tri[1].i,start_tri[1].j,start_tri[1].k);*/

	for (j=0;j<counter;j++){
		start = 1;
		bound_encountered = 0;
		tri = start_tri[j];
		/*tri.i = start_tri[j].i;
		tri.j = start_tri[j].j;
		tri.k = start_tri[j].k;*/
		a = 1.;
		b = 0;
		c = 0;
		if (j==0) add_iso_segment(Res,tri,a,b,c);
		do{	
			center = -1;
			if (fabs(a-1.)<tol) center = tri.i;			
			if (fabs(b-1.)<tol) center = tri.j;
			if (fabs(c-1.)<tol) center = tri.k;	

			if (center>=0 && !start){
				for (i=0;i<Mesh->Sizes[center];i++){
					na1 = Mesh->Connections[center][i];
					if (i==Mesh->Sizes[center]-1){
						if(is_boundary(center)>=0) break; else na2 = Mesh->Connections[center][0];
					}
					else na2 = Mesh->Connections[center][i+1];				
					if (!congruent_triangle(center,na1,na2,tri.i,tri.j,tri.k)){					
						G.x = X[na1]-X[center];
						G.y = X[na2]-X[center];
						New = barycentric_iso_line(0,0,1.0,&G,tol);
						if (New!=NULL){
							a = New[0];
							b = New[1];
							c = New[3];						
							tri.i = na1;
							tri.j = na2;
							list = overlap(na1,na2);
							if (list[0]!=center) tri.k = list[0];
							else if (list[1]>=0 && list[1]!=center) tri.k = list[1];
							else bound_encountered = 1;	
							new_index = tri.k;					
							free(list);
							free(New);
							break;
						}
					}
				}
			}
			else{
				G.x = X[tri.i]-X[tri.k];
				G.y = X[tri.j]-X[tri.k];
				New = barycentric_iso_line(a,b,c,&G,tol);			
				if (New!=NULL){
					a = New[0];
					b = New[1];
					c = New[2];
					if (fabs(New[0])<tol){				
						list = overlap(tri.j,tri.k);
						if (list[0]!=tri.i) tri.i = list[0];
						else if (list[1]>=0 && list[1]!=tri.i) tri.i = list[1];
						else bound_encountered = 1;		
						new_index = tri.i;
						free(list);		
					}
					else if(fabs(b)<tol){
						list = overlap(tri.i,tri.k);
						if (list[0]!=tri.j) tri.j = list[0];
						else if (list[1]>=0 && list[1]!=tri.j) tri.j = list[1];
						else bound_encountered = 1;		
						new_index = tri.j;
						free(list);						
					}
					else if (fabs(c)<tol){
						list = overlap(tri.i,tri.j);
						if (list[0]!=tri.k) tri.k = list[0];
						else if (list[1]>=0 && list[1]!=tri.k) tri.k = list[1];
						else bound_encountered = 1;		
						new_index = tri.k;
						free(list);		
					}
					else{
						printf("error in barycentric_iso_lie: wrong state enceoutnered (%e,%e,%e)-> abort\n",a,b,c);
						exit(0);
					}
					free(New);
				}
				else{
					printf("error in barycentric_iso_lie:no intersection found for triangle (%d,%d,%d) and point (%e,%e,%e)-> abort\n",tri.i,tri.j,tri.k,a,b,c);
					exit(0);
				}								
			}
			add_iso_segment(Res,tri,a,b,c);
			start = 0;
		}while(!bound_encountered && new_index!=start_index && Res->size<max_len);
	
		if (new_index==start_index){		
			add_iso_segment(Res,start_tri[j],1.0,0,0);
			break;
		}
		else if (j==0) first_size = Res->size;		
	}
	
	if (new_index!=start_index && Res->size>first_size){
		index3D* Temp_i = (index3D*)malloc((Res->size)*sizeof(index3D));
		point2D* Temp_c = (point2D*)malloc((Res->size)*sizeof(point2D));		
		memcpy(Temp_i,Res->Tri_indices,(Res->size)*sizeof(index3D));
		memcpy(Temp_c,Res->Coordinates,(Res->size)*sizeof(point2D));
		int s = Res->size-first_size;
		
		for (i=0;i<s;i++){
			Res->Tri_indices[i] = Temp_i[Res->size-i-1];
			Res->Coordinates[i] = Temp_c[Res->size-i-1];
		}
		for (i=0;i<first_size;i++){
			Res->Tri_indices[i+s] = Temp_i[i];
			Res->Coordinates[i+s] = Temp_c[i];
		}
		free(Temp_i);
		free(Temp_c);
		
		/*index3D temp_ind;
		point2D co;
		j = (Res->size-first_size)/2;
		for (i=first_size;i<first_size+j;i++){
			temp_ind = Res->Tri_indices[i];			
			Res->Tri_indices[i] = Res->Tri_indices[Res->size-i-1+first_size];
			Res->Tri_indices[Res->size-i-1+first_size] = temp_ind;
			
			co = Res->Coordinates[i];
			Res->Coordinates[i] = Res->Coordinates[Res->size-i-1+first_size];
			Res->Coordinates[Res->size-i-1+first_size] = co;
		}*/						
	}
	
	return Res;
}

void isoline_to_vertices(mesh2D* Mesh,isoline* Iso,point2D** Vertices,int* size){
	int i;
	double a,b,c;
	index3D ind;
	point2D A,B,C;
	
	*size = Iso->size;
	*Vertices = (point2D*)realloc(*Vertices,(*size)*sizeof(point2D));
	
	for (i=0;i<(*size);i++){
		ind = Iso->Tri_indices[i];
		A = Mesh->Points[ind.i];
		B = Mesh->Points[ind.j];
		C = Mesh->Points[ind.k];
		
		a = Iso->Coordinates[i].x;
		b = Iso->Coordinates[i].y;
		c = 1.-a-b;
		
		(*Vertices)[i].x = a*A.x+b*B.x+c*C.x;
		(*Vertices)[i].y = a*A.y+b*B.y+c*C.y;		
	}
}

void free_isoline(isoline** Iso){
	if (*Iso!=NULL){
		if ((*Iso)->Tri_indices!=NULL) free((*Iso)->Tri_indices);
		if ((*Iso)->Coordinates!=NULL) free((*Iso)->Coordinates);
		free(*Iso);
		*Iso = NULL;
	}
}

void get_iso_start_indices_line(point2D* P1,point2D* P2,int num,int* Indices){
	int i;
	double s,last;
	point2D P;
	
	for (i=0;i<num;i++){
		s = (double)i/(num-1);
		P.x = P1->x+(P2->x-P1->x)*s;
		P.y = P1->y+(P2->y-P1->y)*s;
		Indices[i] = get_nearest_index(&P);
		//printf("index %d at (%f,%f)\n",i,glob_mesh.Points[Indices[i]].x,glob_mesh.Points[Indices[i]].y);
		if (i>0 && Indices[i]==Indices[i-1]){
			printf("Warning in %s: points too close\n",__func__);
		}
	}
}

double interpolate_single_value(double* X,point2D* P){
	int* Ind = get_triangle(P,NULL,DBL_MAX);
	if (Ind!=NULL){
		point2D bar = get_barycentric_2D(&(glob_mesh.Points[Ind[0]]),&(glob_mesh.Points[Ind[1]]),&(glob_mesh.Points[Ind[2]]),P);		
		double res = bar.x*X[Ind[0]]+bar.y*X[Ind[1]]+(1.-bar.x-bar.y)*X[Ind[2]];		
		free(Ind);
		return res;
	}
	else return NAN;
}

void find_iso_start_indices_eq_dist(double* X,point2D* P1,point2D* P2,int num,int* Indices){
	const double tol = 1e-6;
	
	int i,j;
	point2D p;
	double s,v,v1,v2,vl,vr,l,r,m;
	
	v1 = interpolate_single_value(X,P1);
	v2 = interpolate_single_value(X,P2);
	Indices[0] = get_nearest_index(P1);
	Indices[num-1] = get_nearest_index(P2);
	
	printf("\nvalue 0: %e, value %d: %e\n",v1,num-1,v2);
	
	for (i=1;i<num-1;i++){
		v = (double)v1+(v2-v1)*i/(num-1);
		printf("value %d: %e\n",i,v);
		l = 0;
		r = 1.;
		vl = v1;
		vr = v2;
		do{
			s = (l+r)/2.;
			p.x = P1->x+(P2->x-P1->x)*s;
			p.y = P1->y+(P2->y-P1->y)*s;
			m = interpolate_single_value(X,&p);
			if (isnan(m)) break;
			if ((vl-v)*(m-v)<0){
				r = s;
				vr = m;
			}
			else{ //if ((vr-v)*(m-v)<0){
				l = s;
				vl = m;	
			}		
		}while(fabs((vr-vl)/(v2-v1))>tol);
		
		if (!isnan(m)){
			s = (r+l)/2.;
			p.x = P1->x+(P2->x-P1->x)*s;
			p.y = P1->y+(P2->y-P1->y)*s;
			Indices[i] = get_nearest_index(&p);
		}
		else Indices[i] = -1;
		//printf("index %d at (%f,%f)\n",i,glob_mesh.Points[Indices[i]].x,glob_mesh.Points[Indices[i]].y);
	}
	
}

double mean2D(double* X){
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double sum = 0;
	for (i=0;i<n;i++) sum += V[i];
	double res = scalar(X,V,n)/sum;
	free(V);
	return res;
}

double variance2D(double* X){
	int i;
	int n = glob_mesh.size;
	int old = get_var_number2D();
	set_var_number2D(1);
	double* V = set_vector_Ui_0_2D(1);
	sparse_matrix* A = set_matrix_Aij_00_2D(0,0,&insert_AV);
	
	double area = 0;
	for (i=0;i<n;i++) area += V[i];
	double mean = scalar(X,V,n)/area;
	
	double* Y = clone_vector(X,n);
	vector_shift(Y,n,-mean);
	double res = sqrt(sparse_bilinear(Y,A,Y)/area);
	
	set_var_number2D(old);
	free_sparse(A);
	free(V);
	free(Y);
	return res;
}

double moment_2D(double* X,int order){
	int i,j;
	double p;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double area = get_total_area();
	double mean = scalar(X,V,n)/area;
	double sum = 0;
	
	for (i=0;i<n;i++){
		p = 1.;
		for (j=0;j<order;j++) p *= X[i]-mean;
		sum += fabs(p*V[i]);
	}
	free(V);
	return pow(sum/area,(double)1/order);
}

double* cartesian_function_on_mesh2D(double (*Func)(double x,double y)){
	int i;
	double x,y;
	int n = glob_mesh.size;
	if (Func==NULL) return NULL;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		Res[i] = (*Func)(x,y);
	}
	return Res;
}

/*double* hesse(double* Data){
	int n = glob_mesh.size;
	int old = get_var_number2D();
	set_var_number2D(4);
	double* Res = zero_vector(4*n);
	double* V = set_vector_Ui_0_2D(4);
	sparse_matrix* K = set_matrix_Aij_11_2D(0,0,&insert_A_abV);
	copy_vector_content(Data,Res,0,0,n);
	sparse_multiplication(K,Res);
	vector_pseudo_div(Res,V,4*n);
	set_var_number2D(old);
	free_sparse(K);
	free(V);
	return Res;
}*/

index2D get_most_distinct_points(){
	int i,j;
	double d = 0;
	index2D res;
	res.i = 0;
	res.j = 0;
	for (i=0;i<glob_mesh.size;i++){
		for (j=0;j<i;j++){
			if (sqrdist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]))>d){
				res.i = i;
				res.j = j;
			}
		}
	}
	return res;
	
}

double* nodewise_tensor_contract_2D(double* A,double* B){						// R^i = A^i_ab*B^i_ba
	int i;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	for (i=0;i<n;i++){
		Res[i] = A[i]*B[i]+A[n+i]*B[2*n+i]+A[2*n+i]*B[n+i]+A[3*n+i]*B[3*n+i];
	}
	return Res;
}
																						//	 bd     bd
double* sparse3D_bilinear_tensor_2D_right(sparse_matrix3D* B,double* X,double* Y){		// B^i = B^ijk*Xj*Yk
	int i,j,k,l,m,b,d;																	// output has 2^2 components: Gxx->0,Gxy->1,...
	double p;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(4*n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
					
			l = 2*b+d;
			Res[l*n+i] += p*X[j]*Y[k];
		}
	}	
	return Res;
}

																				//	     	bd
sparse_matrix* sparse3D_times_rank2_2D_left(sparse_matrix3D* B,double* G){		// G^i_bd.B^ijk
	int I,i,j,k,l,m,b,d;														// 2^2 components: Gxx->0,Gxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(n);
	
	for (I=0;I<2*n;I++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = I / n;
			i = I % n;
			
			d = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			k = B->Indices2[i][m];
					
			l = 2*b+d;
			insert_sparse(Res,G[n*l+j]*p,i,k);		
		}
	}	
	return Res;
}
																				//	     	 bd
sparse_matrix* sparse3D_times_rank2_2D_right(sparse_matrix3D* B,double* G){		// G^i_bd.B^ijk
	int i,j,k,l,m,b,d;														// 2^2 components: Gxx->0,Gxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
					
			l = 2*b+d;
			insert_sparse(Res,G[n*l+i]*p,j,k);		
		}
	}	
	return Res;
}
																									//		         	   bd		
double* sparse3D_times_rank2_times_scalar_2D_right(sparse_matrix3D* B,double* Gab,double* X){		// R_j = G^i_bd.z^k.B^ijk
	int i,j,k,l,m,b,d;																	
	double p;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
					
			l = 2*b+d;
			Res[j] += Gab[n*l+i]*X[k]*p;
		}
	}	
	return Res;
}

int rank4_map(char I1,char I2,char I3,char I4){
	const int d = 2;
	int i1 = (I1=='x'?0:1);
	int i2 = (I2=='x'?0:1);
	int i3 = (I3=='x'?0:1);
	int i4 = (I4=='x'?0:1);
	return ((i1*d+i2)*d+i3)*d+i4;
}
																						//		             bd
sparse_matrix* sparse3D_times_rank4_2D_middle(sparse_matrix3D* B,double* C,double* U){	// C^i_abcd.U^j_a.B^ijk
	int i,j,k,l,m,a,b,c,d;																// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
			
			a = 0;
			c = 0;
			l = 8*a+4*b+2*c+d;
			//insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			insert_sparse(Res,U[n*a+j]*C[n*l+i]*p,i,n*c+k);
			c = 1;
			l = 8*a+4*b+2*c+d;
			//insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			insert_sparse(Res,U[n*a+j]*C[n*l+i]*p,i,n*c+k);
			a = 1;
			l = 8*a+4*b+2*c+d;
			//insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			insert_sparse(Res,U[n*a+j]*C[n*l+i]*p,i,n*c+k);
			c = 0;
			l = 8*a+4*b+2*c+d;
			//insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);			
			insert_sparse(Res,U[n*a+j]*C[n*l+i]*p,i,n*c+k);
		}
	}	
	return Res;
}
																				//			   bd
sparse_matrix* sparse3D_times_rank4_2D_right(sparse_matrix3D* B,double* C){		// C^i_abcd.B^ijk
	int i,j,k,l,m,a,b,c,d;														// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(2*n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
			
			a = 0;
			c = 0;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			c = 1;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			a = 1;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			c = 0;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);			
		}
	}	
	return Res;
}

void sparse3D_times_rank4_2D_right_get_map(sparse_matrix3D* B,sparse_matrix* A,map_sparse** Map){	
	int i,j,k,m,a,c,counter;															// 2^4 components: Cxxxx->0,C->xxxy->1,...
	
	int n = glob_mesh.size;
	(*Map) = (map_sparse*)malloc(sizeof(map_sparse));
	(*Map)->Indices = NULL;
	counter = 0;
	
	for (i=0;i<n;i++){
		(*Map)->Indices = (int*)realloc((*Map)->Indices,(counter+8*(B->Len[i]))*sizeof(int));
		for (m=0;m<B->Len[i];m++){			
			j = B->Indices1[i][m] % n;
			k = B->Indices2[i][m] % n;
			
			a = 0;
			c = 0;
			(*Map)->Indices[counter++] = n*a+j;
			(*Map)->Indices[counter++] = find_position_in_row(A,n*a+j,n*c+k);
			c = 1;			
			(*Map)->Indices[counter++] = n*a+j;
			(*Map)->Indices[counter++] = find_position_in_row(A,n*a+j,n*c+k);
			a = 1;
			(*Map)->Indices[counter++] = n*a+j;
			(*Map)->Indices[counter++] = find_position_in_row(A,n*a+j,n*c+k);
			c = 0;
			(*Map)->Indices[counter++] = n*a+j;
			(*Map)->Indices[counter++] = find_position_in_row(A,n*a+j,n*c+k);		
		}
	}	
	(*Map)->Pattern = clone(A);
}

sparse_matrix* sparse3D_times_rank4_2D_right_use_map(sparse_matrix3D* B,double* C,map_sparse* Map){	
	int i,j,k,l,m,a,b,c,d,counter;														// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double p;
	
	int n = glob_mesh.size;
	int N = (Map->Pattern)->size;
	sparse_matrix* Res = (sparse_matrix*)malloc(sizeof(sparse_matrix));
	Res->size = N;
	Res->Indices = (int**)malloc(N*sizeof(int*));
	Res->Values = (double**)malloc(N*sizeof(double*));
	Res->Len = zero_int_list(N);
	memcpy(Res->Len,(Map->Pattern)->Len,N*sizeof(int));
	for (i=0;i<N;i++){
		m = (Map->Pattern)->Len[i];
		Res->Indices[i] = zero_int_list(m);
		Res->Values[i] = zero_vector(m);
		memcpy(Res->Indices[i],(Map->Pattern)->Indices[i],m*sizeof(int));
	}
	
	counter = 0;
	int* Iterator = Map->Indices;
	for (k=0;k<n;k++){
		for (m=0;m<B->Len[k];m++){			
			p = B->Values[k][m];
			b = B->Indices1[k][m];
			if (b<n) b = 0; else b = 1;
			d = B->Indices2[k][m];		
			if (d<n) d = 0; else d = 1;
			l = n*(4*b+d)+k;
						
			i = *Iterator;Iterator++;
			j = *Iterator;Iterator++;			
			Res->Values[i][j] += C[l]*p;
						
			l += 2*n;
			i = *Iterator;Iterator++;
			j = *Iterator;Iterator++;
			Res->Values[i][j] += C[l]*p;
						
			l += 8*n;
			i = *Iterator;Iterator++;
			j = *Iterator;Iterator++;			
			Res->Values[i][j] += C[l]*p;
			
			
			l -= 2*n;
			i = *Iterator;Iterator++;
			j = *Iterator;Iterator++;			
			Res->Values[i][j] += C[l]*p;
		}
	}	
	return Res;
}
																				//			  bd
sparse_matrix* sparse3D_times_rank4_2D_left(sparse_matrix3D* B,double* C){		// C^i_abcd.B^ijk
	int h,i,j,k,l,m,a,b,c,d;													// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(2*n);
	
	for (i=0;i<B->size;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = i / n;
			j = i % n;
			
			d = B->Indices1[i][m] / n;
			k = B->Indices1[i][m] % n;
			
			h = B->Indices2[i][m];
			
			a = 0;
			c = 0;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+j]*p,n*a+k,n*c+h);
			c = 1;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+j]*p,n*a+k,n*c+h);
			a = 1;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+j]*p,n*a+k,n*c+h);
			c = 0;
			l = 8*a+4*b+2*c+d;
			insert_sparse(Res,C[n*l+j]*p,n*a+k,n*c+h);			
		}
	}	
	return Res;
}
																					//			   bd
sparse_matrix* sparse3D_times_rank4_2D_right_T(sparse_matrix3D* B,double* C){		// C^i_adcb.B^ijk
	int i,j,k,l,l0,m,a,b,c,d;														// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double p;
	
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(2*n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;
			
			a = 0;
			c = 0;
			l0 = 4*d+b;
			l = l0+8*a+2*c;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			c = 1;
			l = l0+8*a+2*c;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			a = 1;
			l = l0+8*a+2*c;
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);
			c = 0;
			l = l0+8*a+2*c;			
			insert_sparse(Res,C[n*l+i]*p,n*a+j,n*c+k);			
		}
	}	
	return Res;
}
		
double* rank4_2D_first_trace(double* C){
	int i;
	int n = glob_mesh.size;
	double* Res = zero_vector(4*n);
	
	for (i=0;i<n;i++){
		Res[i] = C[i]+C[12*n+i];
		Res[n+i] = C[n+i]+C[13*n+i];
		Res[2*n+i] = C[2*n+i]+C[14*n+i];
		Res[3*n+i] = C[3*n+i]+C[15*n+i];
	}
	return Res;
}

double* rank2_times_rank4_2D(double* M2,double* C){							// computes M2:C4
	int i;
	int n = glob_mesh.size;
	double* Res = zero_vector(4*n);
	
	for (i=0;i<n;i++){
		Res[i] = M2[i]*C[i]+M2[n+i]*C[4*n+i]+M2[2*n+i]*C[8*n+i]+M2[3*n+i]*C[12*n+i];
		Res[n+i] = M2[i]*C[n+i]+M2[n+i]*C[5*n+i]+M2[2*n+i]*C[9*n+i]+M2[3*n+i]*C[13*n+i];
		Res[2*n+i] = M2[i]*C[2*n+i]+M2[n+i]*C[6*n+i]+M2[2*n+i]*C[10*n+i]+M2[3*n+i]*C[14*n+i];
		Res[3*n+i] = M2[i]*C[3*n+i]+M2[n+i]*C[7*n+i]+M2[2*n+i]*C[11*n+i]+M2[3*n+i]*C[15*n+i];		
	}
	return Res;
}

double* sparse3D_times_rank4_matrix_matrix(sparse_matrix3D* B,double* M,double* P,double* C){		
	int i,j,k,m;																	// M^j_ab.C^i_abcd.P^k_cd.B^ijk., no sum over i
	double p,cm,Rxx,Rxy,Ryx,Ryy;													// 2^4 components: Cxxxx->0,C->xxxy->1,...
	double* It_C;
	double* It_M;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
			j = B->Indices1[i][m];	
			k = B->Indices2[i][m];				
			
			It_C = &(C[i]);
			It_M = &(M[j]);
			
			Rxx = (*It_C)*(*It_M);
			It_C += n;
			Rxy = (*It_C)*(*It_M);
			It_C += n;
			Ryx = (*It_C)*(*It_M);
			It_C += n;
			Ryy = (*It_C)*(*It_M);
			It_C += n;
			It_M += n;
			
			Rxx += (*It_C)*(*It_M);
			It_C += n;
			Rxy += (*It_C)*(*It_M);
			It_C += n;
			Ryx += (*It_C)*(*It_M);
			It_C += n;
			Ryy += (*It_C)*(*It_M);
			It_C += n;
			It_M += n;
			
			Rxx += (*It_C)*(*It_M);
			It_C += n;
			Rxy += (*It_C)*(*It_M);
			It_C += n;
			Ryx += (*It_C)*(*It_M);
			It_C += n;
			Ryy += (*It_C)*(*It_M);
			It_C += n;
			It_M += n;
			
			Rxx += (*It_C)*(*It_M);
			It_C += n;
			Rxy += (*It_C)*(*It_M);
			It_C += n;
			Ryx += (*It_C)*(*It_M);
			It_C += n;
			Ryy += (*It_C)*(*It_M);
			
			Res[i] += p*(Rxx*P[k]+Rxy*P[n+k]+Ryx*P[2*n+k]+Rxx*P[3*n+k]);	
		}
	}
	return Res;
}

double* sparse3D_times_rank4_matrix_vector(sparse_matrix3D* B,double* C,double* M,double* X){		
	int i,j,k,m,c,d;																// M^j_ab.C^i_abcd.B^ijk_d.X^k_c, no sum over i
	double p,cm;																	// 2^4 components: Cxxxx->0,C->xxxy->1,...
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
			j = B->Indices1[i][m];	
			k = B->Indices2[i][m];	
		
			d = (k<n) ? 0:1;
			k -= n*d;
			
			c = 0;
			cm = C[(2*c+d)*n+i]*M[j] + C[(4+2*c+d)*n+i]*M[n+j] + C[(8+2*c+d)*n+i]*M[2*n+j] + C[(12+2*c+d)*n+i]*M[3*n+j];
			Res[i] += p*cm*X[n*c+k];
			
			c = 1;
			cm = C[(2*c+d)*n+i]*M[j] + C[(4+2*c+d)*n+i]*M[n+j] + C[(8+2*c+d)*n+i]*M[2*n+j] + C[(12+2*c+d)*n+i]*M[3*n+j];
			Res[i] += p*cm*X[n*c+k];		
		}
	}
	return Res;
}

double* sparse3D_times_rank4_matrix(sparse_matrix3D* B,double* C,double* M){		
	int i,j,k,l,m,c,d;																// M^j_ab.C^k_abcd.B^ijk_d, no sum over i
	double p,cm;																	// 2^4 components: Cxxxx->0,C->xxxy->1,...
	
	int n = glob_mesh.size;
	double* Res = zero_vector(2*n);
	
	for (i=0;i<B->size;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
			j = B->Indices1[i][m];	
			k = B->Indices2[i][m];	
		
			d = (i<n) ? 0:1;
			l = i-d*n;
			
			c = 0;
			cm = C[(2*c+d)*n+k]*M[j] + C[(4+2*c+d)*n+k]*M[n+j] + C[(8+2*c+d)*n+k]*M[2*n+j] + C[(12+2*c+d)*n+k]*M[3*n+j];
			Res[c*n+l] += p*cm;
			
			c = 1;
			cm = C[(2*c+d)*n+k]*M[j] + C[(4+2*c+d)*n+k]*M[n+j] + C[(8+2*c+d)*n+k]*M[2*n+j] + C[(12+2*c+d)*n+k]*M[3*n+j];
			Res[c*n+l] += p*cm;		
		}
	}
	return Res;
}

double* sparse3D_times_rank4_2D_bilinear(sparse_matrix3D* B,double* C,double* X1,double* X2){		
	int i,j,k,l,m,a,b,c,d;												// X1^j_a.C^i_abcd.B^ijk_bd.X2^k_c, no sum over i
	double p;															// 2^4 components: Cxxxx->0,C->xxxy->1,...
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	for (i=0;i<n;i++){
		for (m=0;m<B->Len[i];m++){			
			p = B->Values[i][m];
						
			/*b = B->Indices1[i][m] / n;
			j = B->Indices1[i][m] % n;
			
			d = B->Indices2[i][m] / n;
			k = B->Indices2[i][m] % n;*/
			
			/*a = 0;
			c = 0;
			l = 8*a+4*b+2*c+d;
			Res[i] += C[n*l+i]*p*X1[n*a+j]*X2[n*c+k];
			c = 1;
			l = 8*a+4*b+2*c+d;
			Res[i] += C[n*l+i]*p*X1[n*a+j]*X2[n*c+k];
			a = 1;
			l = 8*a+4*b+2*c+d;
			Res[i] += C[n*l+i]*p*X1[n*a+j]*X2[n*c+k];
			c = 0;
			l = 8*a+4*b+2*c+d;
			Res[i] += C[n*l+i]*p*X1[n*a+j]*X2[n*c+k];*/
			
			l = B->Indices1[i][m];	
			b = (l<n) ? 0:1;
			j = l-b*n;
			
			l = B->Indices2[i][m];	
			d = (l<n) ? 0:1;
			k = l-d*n;
			
			a = j;
			c = k;		
			l = (4*b+d)*n+i;
			Res[i] += C[l]*p*X1[a]*X2[c];
			c += n;
			l += 2*n;
			Res[i] += C[l]*p*X1[a]*X2[c];
			a += n;
			l += 8*n;
			Res[i] += C[l]*p*X1[a]*X2[c];
			c -= n;
			l -= 2*n;
			Res[i] += C[l]*p*X1[a]*X2[c];
		}
	}	
	return Res;
}

double* rank4_unity_2D(){
	int i;
	int n = glob_mesh.size;
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		Res[i] = 1.;
		Res[5*n+i] = 1.;
		Res[10*n+i] = 1.;
		Res[15*n+i] = 1.;
	}
	return Res;
}

sparse_matrix* get_stiffness_tensor(double* Rank4_coeff){
	
	#define C(i,j,k,l) (&(Rank4_coeff[rank4_map(i,j,k,l)*n]))
	
	static sparse_matrix3D* Bxx = NULL;
	static sparse_matrix3D* Bxy = NULL;
	static sparse_matrix3D* Byx = NULL;
	static sparse_matrix3D* Byy = NULL;

	int i,j,k,l;
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(2*n);
	if (Bxx==NULL){
		set_var_number2D(1);
		Bxx = set_matrix_Bijk_011(0,0,0,&insert_WB_xx);
		Bxy = set_matrix_Bijk_011(0,0,0,&insert_WB_xy);
		Byx = set_matrix_Bijk_011(0,0,0,&insert_WB_yx);
		Byy = set_matrix_Bijk_011(0,0,0,&insert_WB_yy);
	}
	
	sparse_matrix* Axx_xx = sparse3D_vB(Bxx,C('x','x','x','x'),n);
	sparse_matrix* Axx_xy = sparse3D_vB(Bxy,C('x','x','x','y'),n);
	sparse_matrix* Axx_yx = sparse3D_vB(Byx,C('x','y','x','x'),n);
	sparse_matrix* Axx_yy = sparse3D_vB(Byy,C('x','y','x','y'),n);
	sparse_add(Axx_xx,Axx_xy,1.);
	sparse_add(Axx_xx,Axx_yx,1.);
	sparse_add(Axx_xx,Axx_yy,1.);
	enlarge_matrix(Axx_xx,n,2*n,0,0);

	sparse_matrix* Axy_xx = sparse3D_vB(Bxx,C('x','x','y','x'),n);
	sparse_matrix* Axy_xy = sparse3D_vB(Bxy,C('x','x','y','y'),n);
	sparse_matrix* Axy_yx = sparse3D_vB(Byx,C('x','y','y','x'),n);
	sparse_matrix* Axy_yy = sparse3D_vB(Byy,C('x','y','y','y'),n);
	sparse_add(Axy_xx,Axy_xy,1.);
	sparse_add(Axy_xx,Axy_yx,1.);
	sparse_add(Axy_xx,Axy_yy,1.);
	enlarge_matrix(Axy_xx,n,2*n,0,1);
	
	sparse_matrix* Ayx_xx = sparse3D_vB(Bxx,C('y','x','x','x'),n);
	sparse_matrix* Ayx_xy = sparse3D_vB(Bxy,C('y','x','x','y'),n);
	sparse_matrix* Ayx_yx = sparse3D_vB(Byx,C('y','y','x','x'),n);
	sparse_matrix* Ayx_yy = sparse3D_vB(Byy,C('y','y','x','y'),n);
	sparse_add(Ayx_xx,Ayx_xy,1.);
	sparse_add(Ayx_xx,Ayx_yx,1.);
	sparse_add(Ayx_xx,Ayx_yy,1.);
	enlarge_matrix(Ayx_xx,n,2*n,1,0);
	
	sparse_matrix* Ayy_xx = sparse3D_vB(Bxx,C('y','x','y','x'),n);
	sparse_matrix* Ayy_xy = sparse3D_vB(Bxy,C('y','x','y','y'),n);
	sparse_matrix* Ayy_yx = sparse3D_vB(Byx,C('y','y','y','x'),n);
	sparse_matrix* Ayy_yy = sparse3D_vB(Byy,C('y','y','y','y'),n);
	sparse_add(Ayy_xx,Ayy_xy,1.);
	sparse_add(Ayy_xx,Ayy_yx,1.);
	sparse_add(Ayy_xx,Ayy_yy,1.);
	enlarge_matrix(Ayy_xx,n,2*n,1,1);
	
	sparse_add(Res,Axx_xx,1.);
	sparse_add(Res,Axy_xx,1.);
	sparse_add(Res,Ayx_xx,1.);
	sparse_add(Res,Ayy_xx,1.);

	free_sparse(Axx_xx);
	free_sparse(Axx_xy);
	free_sparse(Axx_yx);
	free_sparse(Axx_yy);
	free_sparse(Axy_xx);
	free_sparse(Axy_xy);
	free_sparse(Axy_yx);
	free_sparse(Axy_yy);
	free_sparse(Ayx_xx);
	free_sparse(Ayx_xy);
	free_sparse(Ayx_yx);
	free_sparse(Ayx_yy);
	free_sparse(Ayy_xx);
	free_sparse(Ayy_xy);
	free_sparse(Ayy_yx);
	free_sparse(Ayy_yy);
	
	#undef C(i,j,k,l)
	
	return Res;
}

