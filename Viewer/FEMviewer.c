#include <stdio.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <png.h>
#include <dirent.h>

#include "linear_algebra.h"
#include "../FEM2D/FEM2D.h"
#include "../FEM2D/geometry2D.h"

#define KEY_TAB 9

#define RGB 0
#define ALPHA 1
#define WIRES 0
#define FILL 1
#define POINTS 2
#define FAILED 0
#define SUCCESS 1
#define NORMAL 0
#define MAKEMOVIE 1
#define SIMPLEVIEW 2
#define REFERENCE 0
#define DEFORMED 1
#define QUIET 1
#define CHATTY 0
#define AUTOMATIC_RANGE 1
#define MANUAL_RANGE 0
#define LAST_TIME_STEP -1

#define INDEX_DISPLACEMENT 0
#define INDEX_DAMAGE 2


const char* Bin_name = "data.bin";
const float FOV_angle = 45.;
const float FOV_aspect_ratio = 2.;
const float FOV_cutoff_min = 0.02;
const float FOV_cutoof_max = 100.;

const double cam_posx_0 = 0.0;
const double cam_posy_0 = 0.5;
const double cam_posz_0 = 2.5;

const double cracked_tol = 0.01;

static char Home[512];
static char DataFileName[512];
static int draw_mode = FILL;
static int Program_Mode = NORMAL;
static int frame = REFERENCE;
static int range_mode = AUTOMATIC_RANGE;
static int show_damage = 0;
static int show_isolines = 0;
static int show_vectors = 0;

static int selection = 0;
static int selection_frame= 0;
static int selection_deactivated = 0;
static int selection_p1x = 0;
static int selection_p1y = 0;
static int selection_p2x = 0;
static int selection_p2y = 0;

static double glob_angle_z = 0;
static double cam_posz = 0;
static double cam_posx = 0;
static double cam_posy = 0;

static char* Mesh_name = NULL;
static char** Field_Names = NULL;
static int data_field_num = 0;
static int timestep_num = 0;
static int current_timestep = 0;
static int current_field_index = 0;
static int current_node_index = 0;
static int current_mesh_index = 1;
static int man_range_index = 0;
static int step_size = 1;
static int frames_per_sec = 15;
static double man_range_min = NAN;
static double man_range_max = NAN;
static point2D current_mouse_pos;
static double* DataArray = NULL;
static double* Displacement = NULL;
static double* VectorField = NULL;
static double* Damage = NULL;
static double* Glob_Min = NULL;
static double* Glob_Max = NULL;
static double* Times = NULL;
static point2D* Segments = NULL;
static int Segment_num = 0;
extern mesh2D glob_mesh;
extern element_collection elements;
static int* Mesh_sizes = NULL;
static int Mesh_number = 0;
static index2D* time_step_range = NULL;

void freeLine(char*** Line,int len){
	int i;
	for (i=0;i<len;i++) free((*Line)[i]);
	free(*Line);
}

int contains(int* List,int size,int ele){
	int i;
	if (List==NULL) return 0;
	for (i=0;i<size;i++) if (List[i]==ele) return 1;
	return 0;
}

void insert_connection(int node,int i,mesh2D* Mesh){
	if (!contains(Mesh->Connections[node],Mesh->Sizes[node],i)){
		Mesh->Sizes[node]++;
		Mesh->Connections[node] = (int*)realloc(Mesh->Connections[node],Mesh->Sizes[node]*sizeof(int));
		Mesh->Connections[node][Mesh->Sizes[node]-1] = i;
	}
}

double* cross3D(double Ax,double Ay,double Az,double Bx,double By,double Bz){
	double* Res = (double*)malloc(3*sizeof(double));
	Res[0] = Ay*Bz-Az*By;
	Res[1] = Az*Bx-Ax*Bz;
	Res[2] = Ax*By-Ay*Bx;
	return Res;
}

void normalize3D(double* X,double* Y,double* Z){
	double norm = sqrt((*X)*(*X)+(*Y)*(*Y)+(*Z)*(*Z));
	*X /= norm;
	*Y /= norm;
	*Z /= norm;
}

double* Triang_normal(double P1x,double P1y,double P1z,double P2x,double P2y,double P2z,double P3x,double P3y,double P3z){
	double* Res = cross3D(P2x-P1x,P2y-P1y,P2z-P1z,P3x-P1x,P3y-P1y,P3z-P1z);
	normalize3D(&(Res[0]),&(Res[1]),&(Res[2]));
	return Res;
}

void minmax(double* Data,int size,double* Min,double* Max){
	int i;
	double min = Data[0];
	double max = Data[0];
	
	for (i=1;i<size;i++){
		if (Data[i]<min) min = Data[i];
		if (Data[i]>max) max = Data[i];
	}
	*Min = min;
	*Max = max;
}

double vec_norm_2D(double x,double y){
	return sqrt(x*x+y*y);
}

void vec_norm(float x,float y,float* nx,float* ny){
	double d = (float)vec_norm_2D(x,y);
	*nx = x/d;
	*ny = y/d;
}

void vec_ortho_norm(float x,float y,float* nx,float* ny){
	double d = (float)vec_norm_2D(x,y);
	*nx = -y/d;
	*ny = x/d;
}

void minmax_abs_2D(double* Data,int size,double* Min,double* Max){
	int i;
	double s;
	
	if (Data==NULL) return;
	
	int n = glob_mesh.size;
	double min = vec_norm_2D(Data[0],Data[n]);
	double max = vec_norm_2D(Data[0],Data[n]);
	
	for (i=1;i<size;i++){
		s = vec_norm_2D(Data[i],Data[n+i]);
		if (s<min) min = s;
		if (s>max) max = s;
	}
	*Min = min;
	*Max = max;
}

void get_mesh_name(char** MeshName){
	const int BUFF_SIZE = 512;
	
	int i,j,size,number;
	char** Parts;
	char DirName[BUFF_SIZE];
	char Buffer[BUFF_SIZE];
	DIR *dir;
	struct dirent *ent;
	
	sprintf(DirName,"%s/%s/meshes",Home,DataFileName);
	j = 0;
	if ((dir = opendir(DirName)) != NULL){
		while ((ent = readdir(dir)) != NULL){
			sprintf(Buffer,"%s",ent->d_name);
			Parts = split(Buffer,".",&size);
			if (size>0){
				if (j==0){
					*MeshName = (char*)malloc(BUFF_SIZE*sizeof(char));
					sprintf(*MeshName,"%s",Parts[0]);
					j++;
				}
				else{
					if (strcmp(Parts[0],*MeshName)!=0){
						printf("different meshes found: %s %s -> abort\n",Parts[0],*MeshName);
						exit(0);
					}
				}
			}
			for (i=0;i<size;i++) free(Parts[i]);
			free(Parts);
		}
	}
	if (j==0) *MeshName = NULL;
}

void load_mesh(char* Fullname,mesh2D* Mesh,element_collection* Triangles,int quiet){
	const int BUFF_SIZE = 512;
	int i,j,len,n,N,d;
	double x,y;
	index3D ind;
	char Name[BUFF_SIZE];
	char Buffer[BUFF_SIZE];
	char** Line;
	
	// read node file
	sprintf(Name,"%s.node",Fullname);
	FILE* Data = fopen(Name,"r");
	if (Data!=NULL){
		fgets(Buffer,BUFF_SIZE,Data);
		Line = split(Buffer," ",&len);
		if (len<0){
			printf("wrong data file: %s -> abort\n",Name);
			exit(0);
		}
		n = atoi(Line[0]);
		d = atoi(Line[1]);
		if (!quiet){
			printf("loading mesh from file: %s\n",Name);
			printf("dimension: %d\n",d);
			printf("vertices: %d\n",n);
		}
		freeLine(&Line,len);
		Mesh->Points = (point2D*)malloc(n*sizeof(point2D)); 
		Mesh->Connections = (int**)malloc(n*sizeof(int*));
		Mesh->Sizes = (int*)malloc(n*sizeof(int));
		Mesh->Is_boundary = (int*)malloc(n*sizeof(int));
		for (i=0;i<n;i++){
			fgets(Buffer,BUFF_SIZE,Data);
			Line = split(Buffer," ",&len);
			if (len>=4){
				j = atoi(Line[0]);
				Mesh->size = n;
				Mesh->Points[j].x = atof(Line[1]);
				Mesh->Points[j].y = atof(Line[2]);
				Mesh->Is_boundary[j] = atoi(Line[3]);
			}
			else{
				printf("wrong data file -> abort\n");
				exit(0);
			}
			freeLine(&Line,len);
			Mesh->Sizes[i] = 0;
			Mesh->Connections[i] = NULL;
		}
		fclose(Data);
	}
	else{
		printf("could not open file %s\n -> abort\n",Name);
		exit(0);
	}
	
	// read element file
	sprintf(Name,"%s.ele",Fullname);
	Data = fopen(Name,"r");
	if (Data!=NULL){
		fgets(Buffer,BUFF_SIZE,Data);
		Line = split(Buffer," ",&len);
		if (len<0){
			printf("wrong data file: %s -> abort\n",Name);
			exit(0);
		}
		n = atoi(Line[0]);
		if (!quiet) printf("elements: %d\n",n);
		freeLine(&Line,len);
		Triangles->size = n;
		Triangles->Elements = (index3D*)malloc(n*sizeof(index3D)); 
		for (i=0;i<n;i++){
			fgets(Buffer,BUFF_SIZE,Data);
			Line = split(Buffer," ",&len);
			if (len>=4){
				j = atoi(Line[0]);
				Triangles->Elements[j].i = atoi(Line[1]);
				Triangles->Elements[j].j = atoi(Line[2]);
				Triangles->Elements[j].k = atoi(Line[3]);
			}
			else{
				printf("wrong data file -> abort\n");
				exit(0);
			}
			freeLine(&Line,len);
		}
		fclose(Data);
	}
	else{
		printf("could not open file %s\n -> abort\n",Name);
		exit(0);
	}
	
	// get connections
	N = Triangles->size;
	n = Mesh->size;
	for (i=0;i<N;i++){
		ind = Triangles->Elements[i];
		
		insert_connection(ind.i,ind.j,Mesh);
		insert_connection(ind.i,ind.k,Mesh);
		
		insert_connection(ind.j,ind.k,Mesh);
		insert_connection(ind.j,ind.i,Mesh);
		
		insert_connection(ind.k,ind.i,Mesh);
		insert_connection(ind.k,ind.j,Mesh);
	}
	
	// sort knots
	Sort_knots(Mesh);
	
	// create look-up table
	create_look_up_table(Mesh);
}

int Nearest_index(point2D* P){
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

int load_mesh_sizes(int** Sizes,int* len){
	const int BUFF_SIZE = 1024;
	
	char** Parts;
	char Buffer[BUFF_SIZE];
	char Meshname[BUFF_SIZE];
	int i,plen,flag;
	
	int index = 0;
	FILE* Dat = NULL;
	
	*Sizes = NULL;
	*len = 0;
	
	do{
		char Name[BUFF_SIZE];
	
		flag = 0;
		index++;
		//sprintf(Dir,"%s/%s/meshes",Home,DataFileName);
		//fprintf(stdout,"meshes\0");
		sprintf(Name,"%s/%s/meshes/%s.%d.node",Home,DataFileName,Mesh_name,index);
		//printf("open %s\n",Name);
		FILE* Dat = fopen(Name,"r");
		if (Dat!=NULL){
			flag = 1;
			if(fgets(Buffer,BUFF_SIZE,Dat)!=NULL){
				plen = 0;
				Parts = split(Buffer," ",&plen);
				if (plen>0){
					(*len)++;
					*Sizes = (int*)realloc(*Sizes,(*len)*sizeof(int));
					(*Sizes)[(*len)-1] = atoi(Parts[0]);
				}
				//printf("obtain size %d: %s\n",*len-1,Parts[0]);
				for (i=0;i<plen;i++) free(Parts[i]);
				free(Parts);
				//printf("index: %d size: %d\n",index,(*Sizes)[(*len)-1]);
			}
			fclose(Dat);
		}
	}while(flag==1);
	if (*len>0){
		printf("%d meshes found\n",*len);
		return SUCCESS;
	}
	else{
		printf("Warning: could not find meshes in %s/%s/meshes",Home,DataFileName);
		return FAILED;
	}
}

void Load_mesh(int index,mesh2D* Mesh,element_collection* Triangles,int quiet){
	int i,len;
	char Name[512];
	
	sprintf(Name,"%s/%s/meshes/%s.%d",Home,DataFileName,Mesh_name,index);
	load_mesh(Name,Mesh,Triangles,quiet);
}

double* load_data(char* Fullname,int size){
	const int BUFF_SIZE = 512;
	
	int i;
	char Buffer[BUFF_SIZE];
	
	double* Res = zero_vector(size);
	FILE* Data = fopen(Fullname,"r");
	if (Data!=NULL){
		for (i=0;i<size;i++){
			if(fgets(Buffer,BUFF_SIZE,Data)!=NULL) Res[i] = atof(Buffer);
			else{
				printf("line: %d",i);
				printf("wrong data file -> abort\n");
				exit(0);
			}							
		}
	}
	else{
		printf("could not open file %s\n -> abort\n",Fullname);
		exit(0);
	}
	return Res;
}

int binary_load(char* Fullname,double** X,double* time,int* size,int* field_num,int* mesh_index,int index){
	int i,n,m,r;
	FILE* Dat = fopen(Fullname,"r");
	if (Dat!=NULL){
		
		fread(&n,sizeof(int),1,Dat);
		fread(&m,sizeof(int),1,Dat);
		*field_num = n;
		
		//if (index==LAST_INDEX) index = m-1;
		//if (index==PREV_LAST_INDEX) index = m-2;
		
		if (index<0 || index>=m) return FAILED;
		
		// goto index
		for (i=0;i<index;i++){
			fread(time,sizeof(double),1,Dat);
			fread(mesh_index,sizeof(int),1,Dat);
			if (*mesh_index>Mesh_number){
				printf("error in binary load: mesh index %d exceeds number of meshes %d\n -> abort\n",*mesh_index,Mesh_number);
				exit(0);
			}
			fseek(Dat,n*Mesh_sizes[*mesh_index-1]*sizeof(double),SEEK_CUR);
		}
		//fseek(Dat,index*(n+1)*sizeof(double),SEEK_CUR);
		
		// read time
		fread(time,sizeof(double),1,Dat);
		fread(mesh_index,sizeof(int),1,Dat);
		
		// read data
		
		*size = n*Mesh_sizes[*mesh_index-1];
		*X = (double*)malloc((*size)*sizeof(double));
		r = fread(*X,sizeof(double),*size,Dat);
		
		if (n!=0 && r==0){
			printf("error reading file %s -> skip\n",Fullname);
			return FAILED;
		}
		
		int error = 0;
		for (i=0;i<*size;i++) if (isnan((*X)[i])){
			printf("Warning binary_load: NAN encountered at index %d in field %d\n",i % Mesh_sizes[*mesh_index-1],i / Mesh_sizes[*mesh_index-1]);
			error = 1;
		}
		if (error){
			printf("invlaid data -> abort\n");
			exit(0);
		}
		
		// close
		fclose(Dat);
	}
	else{
		printf("Warning: could not open file %s-> abort\n",Fullname);
		exit(0);
	}
	return SUCCESS;	
}

FILE* init_succesive_load(char* Fullname,int* field_num,int start_index,int* max_index){
	int i,n,m,r;
	int index[1];
	double time[1];
	
	FILE* Dat = fopen(Fullname,"r");
	if (Dat!=NULL){		
		fread(&n,sizeof(int),1,Dat);
		fread(max_index,sizeof(int),1,Dat);
		*field_num = n;
		if (start_index>=(*max_index)){
			fclose(Dat);
			return NULL;			
		}
		
		// goto index
		for (i=0;i<start_index;i++){
			fread(time,sizeof(double),1,Dat);
			fread(index,sizeof(int),1,Dat);
			if (index[0]>Mesh_number){
				printf("error in binary load: mesh index %d exceeds number of meshes %d\n -> abort\n",index[0],Mesh_number);
				exit(0);
			}
			fseek(Dat,n*Mesh_sizes[index[0]-1]*sizeof(double),SEEK_CUR);
		}
	}
	return Dat;
}

int succesive_load(FILE* file,int var_num,double** X,double* time,int* size,int* mesh_index){
	int i,n,m,r;
		
	if (file!=NULL){
		fread(time,sizeof(double),1,file);
		fread(mesh_index,sizeof(int),1,file);
		
		*size = var_num*Mesh_sizes[*mesh_index-1];
		*X = (double*)malloc((*size)*sizeof(double));
		r = fread(*X,sizeof(double),*size,file);
		
		if (r==0){
			printf("succesive load: error reading file -> skip\n");
			return FAILED;
		}
		
		int error = 0;
		for (i=0;i<*size;i++) if (isnan((*X)[i])){
			printf("Warning binary_load: NAN encountered at index %d in field %d\n",i % Mesh_sizes[*mesh_index-1],i / Mesh_sizes[*mesh_index-1]);
			error = 1;
		}
		if (error){
			printf("invlaid data -> abort\n");
			exit(0);
		}
		return SUCCESS;
	}
	else return FAILED;
}

void local_minmax(double* Field_data,int field_index){
	int n = Mesh_sizes[current_mesh_index-1];
	minmax(Field_data,n,&(Glob_Min[field_index]),&(Glob_Max[field_index]));
}

void total_minmax(char* Fullname,double** Min,double** Max,double** Times,int* time_num,int* var_num){
	const double tol = 1e-10;
	
	int k,n,i,status,size,mesh_index,max_ind;
	double t,min,max;
	double* X;
	
	if (time_step_range!=NULL) k=time_step_range->i; else k = 0;
	//status = binary_load(Fullname,&X,&t,&size,var_num,&mesh_index,k);
	FILE* Dat = init_succesive_load(Fullname,var_num,k,&max_ind);
	//if (status==FAILED){
	if (Dat==NULL){
		printf("empty data file %s -> abort",Fullname);
		exit(0);
	}
	/*n = Mesh_sizes[mesh_index-1];
	if (size % n !=0){
		printf("wrong mesh size -> abort\n");
		exit(0);
	}*/
	*Times = generate_vector(1,t);
	//*Min = zero_vector(*var_num);
	//*Max = zero_vector(*var_num);
	*Min = generate_vector(*var_num,DBL_MAX);
	*Max = generate_vector(*var_num,-DBL_MAX);	
	//for (i=0;i<(*var_num);i++) minmax(&(X[i*n]),n,&((*Min)[i]),&((*Max)[i]));
	//free(X);	
	
	do{
		if (time_step_range!=NULL && k>time_step_range->j) status = FAILED; 
		//else status = binary_load(Fullname,&X,&t,&size,var_num,&mesh_index,k);
		else if (k>=max_ind) status=FAILED;
		else status = succesive_load(Dat,*var_num,&X,&t,&size,&mesh_index);
		if (status==SUCCESS){
			//printf("load mesh index: %d of %d\n",mesh_index-1,Mesh_number);
			//fflush(stdout);
			n = Mesh_sizes[mesh_index-1];
			for (i=0;i<(*var_num);i++){
				 minmax(&(X[i*n]),n,&min,&max);
				 if (min<(*Min)[i]) (*Min)[i] = min;
				 if (max>(*Max)[i]) (*Max)[i] = max;
			}
			k++;
			*Times = (double*)realloc(*Times,k*sizeof(double));
			(*Times)[k-1] = t;
			free(X);
		}
	}
	while(status==SUCCESS);
	if (Dat!=NULL) fclose(Dat);
	
	*time_num = k;
	printf("%d variables loaded\n",*var_num);
	printf("%d timesteps\n",k);
	
	for (i=0;i<(*var_num);i++) if (fabs((*Min)[i]-(*Max)[i])<tol){
		if ((*Min)[i]!=0) (*Max)[i] = 2.*(*Min)[i]; else (*Max)[i] = 1.;
	}
	
	if (range_mode==MANUAL_RANGE){
		if (man_range_index<(*var_num)){
			if (!isnan(man_range_min)) (*Min)[man_range_index] = man_range_min;
			if (!isnan(man_range_max)) (*Max)[man_range_index] = man_range_max;
		}
		else printf("manual range index out of range -> skip\n");
	}
	for (i=0;i<(*var_num);i++) printf("field #%d: min=%e max=%e\n",i,(*Min)[i],(*Max)[i]);
	if (current_timestep==LAST_TIME_STEP) current_timestep = *time_num-1;
}

double* compute_gradient(double* H){
	int n = glob_mesh.size;
	
	set_var_number2D(2);
	sparse_matrix* Grad = set_matrix_Aij_01_2D(0,0,&insert_A_aV);
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	
	double* GradH = sparse_mult(Grad,H);
	double* X = clone_vector(&(GradH[0]),n);
	double* Y = clone_vector(&(GradH[n]),n);
	
	/*double* V = set_vector_Ui_0_2D(1);
	vector_pseudo_div(GradH,V,n);
	vector_pseudo_div(&(GradH[n]),V,n);*/
	
	double* Sol = zero_vector(n);
	SOR(ID,X,Sol,0.8,-100);
	copy_vector_content(Sol,GradH,0,0,n);
	
	SOR(ID,Y,Sol,0.8,-100);
	copy_vector_content(Sol,GradH,0,n,n);
	
	free_sparse(Grad);
	free_sparse(ID);
	free(Sol);
	free(X);
	free(Y);
	
	return GradH;
}

void get_all_isolines(){
	const double tol = 1e-8;
	const int max_len = 100000;
	const int number = 10;
	point2D P1 = {.x=0.001,.y=0.426};
	point2D P2 = {.x=0.025,.y=0.426};
	
	int i,j;
	
	if (Segments!=NULL){
		free(Segments);
		Segments = NULL;
	}
	
	//printf("creating segments ...\n");
	int* List = zero_int_list(number);
	//get_iso_start_indices_line(&P1,&P2,number,List);
	find_iso_start_indices_eq_dist(DataArray,&P1,&P2,number,List);
	
	for (i=0;i<number;i++){
		int size = 0;
		point2D* Vertices = NULL;
		isoline* Iso = get_isoline(&glob_mesh,DataArray,List[i],tol,max_len);		
		isoline_to_vertices(&glob_mesh,Iso,&Vertices,&size);
		
		int start = Segment_num;
		Segment_num += size-1;
		//printf("segment %d: size=%d\n",i,size);
		Segments = (point2D*)realloc(Segments,2*Segment_num*sizeof(point2D));
		for (j=0;j<size-1;j++){
			Segments[2*(start+j)] = Vertices[j];
			Segments[2*(start+j)+1] = Vertices[j+1];
		}
		
		if (i==number-1){
			Segment_num++;
			Segments = (point2D*)realloc(Segments,2*Segment_num*sizeof(point2D));
			Segments[2*(Segment_num-1)] = Vertices[size-1];
			Segments[2*(Segment_num-1)+1] = Vertices[0];
		}
					
		free_isoline(&Iso);
		free(Vertices);
	}	
	free(List);
}

void get_structure_matrix(double* Matrix,double* Field){
	
	int n = glob_mesh.size;
	int old = get_var_number2D();
	set_var_number2D(4);
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_abV);
	double* AF = sparse_mult(A,Field);
	
	Matrix[0] = scalar(Field,&(AF[0]),n);
	Matrix[1] = scalar(Field,&(AF[n]),n);
	Matrix[2] = scalar(Field,&(AF[2*n]),n);
	Matrix[3] = scalar(Field,&(AF[3*n]),n);

	set_var_number2D(old);
	free_sparse(A);
	free(AF);
}

void load_data_at_time(int time_index,int field_index){
	int size,var_num;
	double t;
	double* Data;
	char Name[512];
	
	if (glob_mesh.Sizes!=NULL) FreeMesh(&glob_mesh);
	if (elements.Elements!=NULL) free(elements.Elements);
	
	sprintf(Name,"%s/%s/%s",Home,DataFileName,Bin_name);
	int status = binary_load(Name,&Data,&t,&size,&var_num,&current_mesh_index,time_index);
	int n = Mesh_sizes[current_mesh_index-1];
	Load_mesh(current_mesh_index,&glob_mesh,&elements,QUIET);
	
	if (status==SUCCESS){
		if (Displacement!=NULL) free(Displacement);
		if (Damage!=NULL) free(Damage);
		if (DataArray!=NULL) free(DataArray);
		Displacement = clone_vector(&(Data[INDEX_DISPLACEMENT*n]),2*n);
		Damage = clone_vector(&(Data[INDEX_DAMAGE*n]),n);
		DataArray = clone_vector(&(Data[field_index*n]),n);
		if (show_isolines) get_all_isolines();
				
		free(Data);
		if (Field_Names!=NULL) glutSetWindowTitle(Field_Names[field_index]);
	}
	else printf ("Warning: could not load data of time index %d field #%d\n",time_index,field_index);
}

void setRGB(png_bytep Dest,float* RGB_element){
	const int RANGE = 255;
	
	float r = roundf((float)RANGE*RGB_element[0]);
	float g = roundf((float)RANGE*RGB_element[1]);
	float b = roundf((float)RANGE*RGB_element[2]);
	Dest[0] = (png_byte)r;
	Dest[1] = (png_byte)g;
	Dest[2] = (png_byte)b;
}

void save_selection(){
	char Name[512];
	sprintf(Name,"%s/%s/selection",Home,DataFileName);	
	FILE* file = fopen(Name,"w");
	if (file!=NULL){
		fprintf(file,"%d\t%d\n",selection_p1x,selection_p1y);
		fprintf(file,"%d\t%d\n",selection_p2x,selection_p2y);
		fclose(file);
	}
	else printf("could not save selction to %s -> skip\n",Name);
}

int load_selection(){
	int n;
	char* part;
	char Name[512];
	
	size_t buff_size = 0;
	char* Buffer = NULL;

	sprintf(Name,"%s/%s/selection",Home,DataFileName);	
	FILE* file = fopen(Name,"r");
	if (file!=NULL){
		getline(&Buffer,&buff_size,file);
		part = strtok(Buffer,"\n\t");
		selection_p1x = atoi(part);
		part = strtok(NULL,"\n\t");
		selection_p1y = atoi(part);
		
		getline(&Buffer,&buff_size,file);
		part = strtok(Buffer,"\n\t");
		selection_p2x = atoi(part);
		part = strtok(NULL,"\n\t");
		selection_p2y = atoi(part);
		
		if (Buffer!=NULL) free(Buffer);
		return 1;
	}
	else return 0;
}

int save_screen(char* Fullname){
	int i,j,x,y,width,height;
	FILE* Imagefile;
	png_structp png_ptr;
	png_infop info_ptr;
	
	if (!selection_deactivated){
		width = abs(selection_p2x-selection_p1x);
		height = abs(selection_p2y-selection_p1y);
		x = (selection_p1x<selection_p2x) ? selection_p1x : selection_p2x;
		y = (selection_p1y<selection_p2y) ? selection_p1y : selection_p2y;
	}
	else{
		width = glutGet(GLUT_WINDOW_WIDTH);
		height = glutGet(GLUT_WINDOW_HEIGHT);
		x = 0;
		y = 0;
	}
	
	float* Pixeldata = (float*)malloc(3*width*height*sizeof(float));
	
	glReadPixels(x,y,width,height,GL_RGB,GL_FLOAT,Pixeldata);
	
	// initialize image structure
	Imagefile = fopen(Fullname,"wb");
	if (Imagefile == NULL){
		printf("could not create file %s -> skip\n",Fullname);
		return 0;
	}
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
	info_ptr = png_create_info_struct(png_ptr);
	if (png_ptr==NULL || info_ptr==NULL){
		printf("failed to initialize png -> skip\n");
		return 0;
	}
	
	// settings
	png_init_io(png_ptr,Imagefile);
	//png_set_filter(png_ptr,0,PNG_FILTER_NONE | PNG_FILTER_SUB | PNG_FILTER_PAETH);
	png_set_IHDR(png_ptr,info_ptr,width,height,8,PNG_COLOR_TYPE_RGB,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_BASE,
		PNG_FILTER_TYPE_BASE);
    /*png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = "Title";
    title_text.text = "title";
    png_set_text(png_ptr, info_ptr, &title_text, 1);*/
	png_write_info(png_ptr,info_ptr);
	
	// write pixeldata to image
	png_bytep row = (png_bytep) malloc(3*width*sizeof(png_byte));
	for (i=0;i<height;i++){
		for (j=0;j<width;j++){
			setRGB(&(row[3*j]),&(Pixeldata[3*((height-i-1)*width+j)]));
		}
		png_write_row(png_ptr,row);
    }
    png_write_end(png_ptr,NULL);
    free(row);
	
	png_free_data(png_ptr,info_ptr,PNG_FREE_ALL,-1);
	png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
	fclose(Imagefile);
	free(Pixeldata);
	return 1;
}

double No_Height(int i,double x,double y){
	return 0;
}

char* FillZero(int i,int max){
	int k;
	char Max[64];
	char* Res = malloc(64*sizeof(char));
	sprintf(Max,"%d",max);
	int len = strlen(Max);
	sprintf(Res,"%d",i);
	int l = strlen(Res);
	if (len!=l){
		for (k=l+1;k>=0;k--) if (k+len-l<64) Res[k+len-l] = Res[k];
		for (k=0;k<len-l;k++) Res[k] = '0';
	}
	return Res;
}

double MapTo01(double val){
	double min = Glob_Min[current_field_index];
	double max = Glob_Max[current_field_index];
	return (val-min)/(max-min);
}

double* RedCMap(double val){
	double* Res = malloc(3*sizeof(double));
	Res[0] = MapTo01(val);
	Res[1] = 0;
	Res[2] = 0;
	return Res;
}

void Rainbow_Map(double* List,double val){
	int i;
	double r,g,b,y;
	y = exp(-20.*(val-0.8)*(val-0.8))/1.5;	
	List[0] = exp(-10.*(val-1.0)*(val-1.0))+exp(-30.*val*val)/2.+y;
	List[1] = exp(-15.*(val-0.5)*(val-0.5))+y;
	List[2] = exp(-10.*(val-0.2)*(val-0.2));
	for (i=0;i<3;i++) if (List[i]>1.) List[i] = 1.;
}

double* Rainbow_CMap(double val){
	const int SIZE = 5000;
	static double* Lookup = NULL;
	int i;
	double x;
	
	// init: create lookup table
	if (Lookup==NULL){		
		fflush(stdout);
		Lookup = zero_vector(3*SIZE);
		for (i=0;i<SIZE;i++){
			x = (double)i/(SIZE-1);
			Rainbow_Map(&(Lookup[3*i]),x);
		}
	}
	
	// default
	double* Res = (double*)malloc(3*sizeof(double));
	x = MapTo01(val);
	i = (int)round((double)(SIZE-1)*x);
	
	if (i<0 || i>=SIZE){
		if (i<0){
			Res[0] = 0;
			Res[1] = 0;
			Res[2] = 0;
		}
		else{
			Res[0] = 1.;
			Res[1] = 1.;
			Res[2] = 1.;
		}
	}
	else{
		Res[0] = Lookup[3*i];
		Res[1] = Lookup[3*i+1];
		Res[2] = Lookup[3*i+2];
	}
	return Res;
}

int MouseToWorld(int winx,int winy,double* X,double* Y,double z){
	GLint status1,status2;
	GLdouble winz,x1,y1,z1,x2,y2,z2;
	GLdouble Model_Matrix[16];
	GLdouble Projection_Matrix[16];
	GLint Viewport[4];
	
	glGetDoublev(GL_MODELVIEW_MATRIX,Model_Matrix);
    glGetDoublev(GL_PROJECTION_MATRIX,Projection_Matrix);
    glGetIntegerv(GL_VIEWPORT,Viewport);
    
    winz = 0.0;
    status1 = gluUnProject(winx,winy,winz,Model_Matrix,Projection_Matrix,Viewport,&x1,&y1,&z1);
	winz = 1.0;
	status2 = gluUnProject(winx,winy,winz,Model_Matrix,Projection_Matrix,Viewport,&x2,&y2,&z2);
    
    if (status1==GL_FALSE || status2==GL_FALSE){
		printf("could not process mouse position at x:%d y:%d\n",winx,winy);
		return FAILED;
	}
	
	double s = (double)(z-z1)/(z2-z1);
	*X = (double)x1+(x2-x1)*s;
	*Y = (double)y1+(y2-y1)*s;
    
    return SUCCESS;
}

float* get_normal_field(double (*Height)(int i,double x,double y)){
	const int SIZE = 9;
	int i;
	double x,y,h1,h2,h3;
	double* H;
	index3D ind;
	point2D* P1;
	point2D* P2;
	point2D* P3;
	int N = elements.size;
	int n = glob_mesh.size;
	float* Res = (float*)malloc(SIZE*N*sizeof(float));
	for (i=0;i<N;i++){
		ind = elements.Elements[i];
		P1 = &(glob_mesh.Points[ind.i]);
		P2 = &(glob_mesh.Points[ind.j]);
		P3 = &(glob_mesh.Points[ind.k]);
		h1 = (*Height)(ind.i,P1->x,P1->y);
		h2 = (*Height)(ind.j,P2->x,P2->y);
		h3 = (*Height)(ind.k,P3->x,P3->y);
		H = Triang_normal(P1->x,P1->y,h1,P2->x,P2->y,h2,P3->x,P3->y,h3);
		
		Res[SIZE*i+0] += (float)H[0];
		Res[SIZE*i+1] += (float)H[1];
		Res[SIZE*i+2] += (float)H[2];
		
		Res[SIZE*i+3] += (float)H[0];
		Res[SIZE*i+4] += (float)H[1];
		Res[SIZE*i+5] += (float)H[2];
				
		Res[SIZE*i+6] += (float)H[0];
		Res[SIZE*i+7] += (float)H[1];
		Res[SIZE*i+8] += (float)H[2];
		
		/*Res[3*ind.i] += H[0];
		Res[3*ind.i+1] += H[1];
		Res[3*ind.i+2] += H[2];
		
		Res[3*ind.j] += H[0];
		Res[3*ind.j+1] += H[1];
		Res[3*ind.j+2] += H[2];
		
		Res[3*ind.k] += H[0];
		Res[3*ind.k+1] += H[1];
		Res[3*ind.k+2] += H[2];*/
		
		free(H);
	}
	
	/*for (i=0;i<n;i++){
		normalize3D(&(Res[3*i]),&(Res[3*i+1]),&(Res[3*i+2]));
		if (Res[3*i+2]<0) Res[3*i+2] *= -1;
	}*/
	return Res;
}

float* ToVertexTriangleArray(double (*Height)(int i,double x,double y)){
	const int SIZE = 9;
	int i;
	double x,y;
	index3D ind;
	int n = glob_mesh.size;
	int N = elements.size;
	float* Res = (float*)malloc(SIZE*N*sizeof(float));
	for (i=0;i<N;i++){
		ind = elements.Elements[i];
		x = glob_mesh.Points[ind.i].x;
		y = glob_mesh.Points[ind.i].y;
		if (frame==DEFORMED){
			x += Displacement[ind.i];
			y += Displacement[n+ind.i];
		}
		Res[SIZE*i] = (float)x;
		Res[SIZE*i+1] = (float)y;
		Res[SIZE*i+2] = (float)(*Height)(ind.i,x,y);
		
		x = glob_mesh.Points[ind.j].x;
		y = glob_mesh.Points[ind.j].y;
		if (frame==DEFORMED){
			x += Displacement[ind.j];
			y += Displacement[n+ind.j];
		}
		Res[SIZE*i+3] = (float)x;
		Res[SIZE*i+4] = (float)y;
		Res[SIZE*i+5] = (float)(*Height)(ind.j,x,y);
		
		x = glob_mesh.Points[ind.k].x;
		y = glob_mesh.Points[ind.k].y;
		if (frame==DEFORMED){
			x += Displacement[ind.k];
			y += Displacement[n+ind.k];
		}
		Res[SIZE*i+6] = (float)x;
		Res[SIZE*i+7] = (float)y;
		Res[SIZE*i+8] = (float)(*Height)(ind.k,x,y);
	}
	return Res;
}

int* ToIndexArray(){
	int i;
	int* Res = (int*)malloc(3*elements.size*sizeof(int));
	for (i=0;i<elements.size;i++){
		Res[3*i] = elements.Elements[i].i;
		Res[3*i+1] = elements.Elements[i].j;
		Res[3*i+2] = elements.Elements[i].k;
	}
	return Res;
}

float* color_map(double* Data,double* (*Color_func)(double val),int withAlpha){	
	int i,j,error;
	index3D ind;
	double* Colors;
	int k = 3+withAlpha;
	int offset = k;
	int N = elements.size;
	float* Res = (float*)malloc(3*k*N*sizeof(float));
	
	/*for (i=0;i<Mesh_sizes[current_mesh_index-1];i++){
		if (isnan(Data[i])){
			printf("bad data at index %d\n",i);		
		}
	}*/
	
	for (i=0;i<N;i++){
		ind = elements.Elements[i];
		
		Colors = (*Color_func)(Data[ind.i]);
		if (show_damage && Damage[ind.i]<cracked_tol) for (j=0;j<3;j++) Res[3*k*i+j] = 0.15;
		else for (j=0;j<3;j++) Res[3*k*i+j] = (float)Colors[j];
		if (withAlpha){
			if (Damage!=NULL) Res[3*k*i+3] = Damage[ind.i]; else Res[3*k*i+3] = 1.f;
		}
		free(Colors);
		
		Colors = (*Color_func)(Data[ind.j]);
		if (show_damage && Damage[ind.j]<cracked_tol) for (j=0;j<3;j++) Res[3*k*i+offset+j] = 0.15;
		else for (j=0;j<3;j++) Res[3*k*i+offset+j] = (float)Colors[j];
		if (withAlpha){
			if (Damage!=NULL) Res[3*k*i+offset+3] = Damage[ind.i]; else Res[3*k*i+offset+3] = 1.f;
		}
		free(Colors);
		
		Colors = (*Color_func)(Data[ind.k]);
		if (show_damage && Damage[ind.k]<cracked_tol) for (j=0;j<3;j++) Res[3*k*i+2*offset+j] = 0.15;
		else for (j=0;j<3;j++) Res[3*k*i+2*offset+j] = (float)Colors[j];
		if (withAlpha){
			if (Damage!=NULL) Res[3*k*i+2*offset+3] = Damage[ind.i]; else Res[3*k*i+2*offset+3] = 1.f;
		}
		free(Colors);
	}
	return Res;
}

void init_buffers(float* Vertices,int* Indices,float* Colors,int withAlpha){
	const int DIM = 3;
	const int VNUM = 3;
	int n = glob_mesh.size;
	int N = elements.size;
	/*GLuint handle_v,handle_i,handle_c;
	
	glGenBuffers(1,&handle_v);
	glBindBuffer(GL_ARRAY_BUFFER,handle_v);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(DIM,GL_FLOAT,0,0);
	
	glGenBuffers(1,&handle_c);
	glBindBuffer(GL_ARRAY_BUFFER,handle_c);
	attribute_v_color = glGetAttribLocation(program,"v_color");
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3+withAlpha,GL_FLOAT,0,0);
	
	glGenBuffers(1,&handle_i);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,handle_i);
	
	glBufferData(GL_ARRAY_BUFFER,DIM*n*sizeof(float),Vertices,GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER,(3+withAlpha)*n*sizeof(float),Colors,GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,VNUM*N*sizeof(int),Indices,GL_STATIC_DRAW);*/
	
	//glDrawElements(GL_TRIANGLES,3*N,GL_UNSIGNED_INT,0);
	
}

float* ColorLegendVertices(float width,float height,int number){
	const int SIZE = 12;
	int i;
	float d = (float)height/number;
	float th = (float)height/100.;
	float* Res = (float*)malloc(SIZE*(number+2)*sizeof(float));
	for (i=0;i<number;i++){
		Res[i*SIZE] = 0;
		Res[i*SIZE+1] = i*d;
		Res[i*SIZE+2] = 0;
		
		Res[i*SIZE+3] = width;
		Res[i*SIZE+4] = i*d;
		Res[i*SIZE+5] = 0;
		
		Res[i*SIZE+6] = width;
		Res[i*SIZE+7] = (i+1)*d;
		Res[i*SIZE+8] = 0;
		
		Res[i*SIZE+9] = 0;
		Res[i*SIZE+10] = (i+1)*d;
		Res[i*SIZE+11] = 0;
	}
	
	// bottom,top bar
	Res[number*SIZE] = -width/2.;
	Res[number*SIZE+1] = -th;
	Res[number*SIZE+2] = 0;
		
	Res[number*SIZE+3] = width;
	Res[number*SIZE+4] = -th;
	Res[number*SIZE+5] = 0;
		
	Res[number*SIZE+6] = width;
	Res[number*SIZE+7] = 0;
	Res[number*SIZE+8] = 0;
		
	Res[number*SIZE+9] = -width/2.;
	Res[number*SIZE+10] = 0;
	Res[number*SIZE+11] = 0;
	
	Res[(number+1)*SIZE] = -width/2.;
	Res[(number+1)*SIZE+1] = number*d;
	Res[(number+1)*SIZE+2] = 0;
		
	Res[(number+1)*SIZE+3] = width;
	Res[(number+1)*SIZE+4] = number*d;
	Res[(number+1)*SIZE+5] = 0;
		
	Res[(number+1)*SIZE+6] = width;
	Res[(number+1)*SIZE+7] = number*d+th;
	Res[(number+1)*SIZE+8] = 0;
		
	Res[(number+1)*SIZE+9] = -width/2.;
	Res[(number+1)*SIZE+10] = number*d+th;
	Res[(number+1)*SIZE+11] = 0;
	
	return Res;
}

float* ColorLegendColors(int number,double* (*Color_func)(double val)){
	const int SIZE = 12;
	int i;
	double h,x,min,max,v;
	double* Color;
	float* Res = (float*)malloc(SIZE*(number+2)*sizeof(float));
	for (i=0;i<number;i++){
		min = Glob_Min[current_field_index];
		max = Glob_Max[current_field_index];
		
		x = (double)i/number;
		v = min+x*(max-min);
		Color = (*Color_func)(v);
		Res[i*SIZE] = Color[0];
		Res[i*SIZE+1] = Color[1];
		Res[i*SIZE+2] = Color[2];
		Res[i*SIZE+3] = Color[0];
		Res[i*SIZE+4] = Color[1];
		Res[i*SIZE+5] = Color[2];
		free(Color);
		
		x = (double)(i+1)/number;
		v = min+x*(max-min);
		Color = (*Color_func)(v);
		Res[i*SIZE+6] = Color[0];
		Res[i*SIZE+7] = Color[1];
		Res[i*SIZE+8] = Color[2];
		Res[i*SIZE+9] = Color[0];
		Res[i*SIZE+10] = Color[1];
		Res[i*SIZE+11] = Color[2];
		free(Color);
	}
	
	// color of bottom,top bar
	for (i=0;i<2*SIZE;i++) Res[number*SIZE+i] = 0;
	
	return Res;
}
	
void draw_object1(){
	/*glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glRotated(glob_angle,0,1.,0);
	
	glMatrixMode(GL_MODELVIEW);*/
	glPushMatrix();	
	glColor3d(1.,0,0);
	glTranslated(1.,0,0);
	glutSolidSphere(.5,20,20);
	glTranslated(.25,0,0);
	glutSolidSphere(.5,20,20);
	glPopMatrix();
}

void init_scene(){
	
	float bg_red, bg_green, bg_blue;
	
	float bg_alpha = 0.0;
	if (draw_mode==FILL){
		bg_red = 0.9;
		bg_green = 0.9;
		bg_blue = 0.9;
	}
	else{
		bg_red = 0;
		bg_green = 0;
		bg_blue = 0;
	}
	
	double upx = 0;
	double upy = 1.;
	double upz = 0;
	
	double center_x = cam_posx;
	double center_y = cam_posy;
	double center_z = 0;
	
	double posx = (cam_posx-center_x)*cos(glob_angle_z)+(cam_posz-center_z)*sin(glob_angle_z)+center_x;
	double posy = cam_posy;
	double posz = -(cam_posx-center_x)*sin(glob_angle_z)+(cam_posz-center_z)*cos(glob_angle_z)+center_z;
	
	glClearColor(bg_red,bg_green,bg_blue,bg_alpha);							// set white background
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOV_angle,FOV_aspect_ratio,FOV_cutoff_min,FOV_cutoof_max);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(posx,posy,posz,center_x,center_y,center_z,upx,upy,upz);
}

/*void init_lighting(){
	GLfloat Light_color_d[] = {0.5,0.5,0.5,1.0};
	GLfloat Light_color_a[] = {0.1,0.1,0.1,1.0};
	//GLfloat Light_position[] = {-4.0,.0,6.0,1.0};			// specify position
	GLfloat Light_position[] = {1,-0.2,0,0};				// specify direction
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,Light_color_d);
	glLightfv(GL_LIGHT0,GL_AMBIENT,Light_color_a);
	glLightfv(GL_LIGHT0,GL_POSITION,Light_position);
}*/

void init_lighting(){
	GLfloat Light_color_d[] = {0.5,0.5,0.5,1.0};
	GLfloat Light_color_a[] = {0.1,0.1,0.1,1.0};
	GLfloat Light_position[] = {0,0,1,0.0};
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,Light_color_d);
	glLightfv(GL_LIGHT0,GL_AMBIENT,Light_color_a);
	glLightfv(GL_LIGHT0,GL_POSITION,Light_position);
}

void draw_legends(){
	const float factor = 0.0005;			// affects text size
	const float linewidth = 3.0;
	const int quadnum = 100;
	const double bar_width = 0.2;
	const double bar_height = 0.6;
	
	float color[] = {0,0,0};
	if (draw_mode!=FILL){
		color[0] = 1.;
		color[1] = 1.;
		color[2] = 1.;
	}
	
	float old_width,bottom,left,dist;
	char ToDraw[32];
	//int len = strlen(ToDraw);
	glGetFloatv(GL_LINE_WIDTH,&old_width);
	glLineWidth(linewidth);
	
	if (show_vectors){
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
	}
	
	if (Program_Mode!=MAKEMOVIE){
		glPushMatrix();
		
		sprintf(ToDraw,"node index=%d",current_node_index);
		glColor3f(color[0],color[1],color[2]);
		glScaled(factor,factor,factor);
		glTranslated(-tan(FOV_angle)*cam_posz_0/(2.0*factor)+(cam_posx-cam_posx_0)/factor,
			-tan(FOV_angle)*cam_posz_0/(6.*FOV_aspect_ratio*factor)+(cam_posy-cam_posy_0)/factor,
			(cam_posz-cam_posz_0-0.1)/factor);
		glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
	
		sprintf(ToDraw,"position x=%1.4f y=%1.4f",current_mouse_pos.x,current_mouse_pos.y);
		glTranslated(-0.13*tan(FOV_angle)*cam_posz_0/factor,-0.02*tan(FOV_angle)*cam_posz_0/factor,0);
		glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
	
		sprintf(ToDraw,"value=%1.10e",DataArray[current_node_index]);
		glTranslated(-0.2*tan(FOV_angle)*cam_posz_0/factor,-0.02*tan(FOV_angle)*cam_posz_0/factor,0);
		glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
		
		glPopMatrix();
	}
	
	
	glPushMatrix();
	sprintf(ToDraw,"time=%.8f",Times[current_timestep]);
	glColor3f(color[0],color[1],color[2]);
	glScaled(factor,factor,factor);
	glTranslated(tan(FOV_angle)*cam_posz_0/(2.6*factor)+(cam_posx-cam_posx_0)/factor,
		-tan(FOV_angle)*cam_posz_0/(6.*FOV_aspect_ratio*factor)+(cam_posy-cam_posy_0)/factor,
		(cam_posz-cam_posz_0-0.1)/factor);
	glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
	glPopMatrix();
	
	glPushMatrix();
	sprintf(ToDraw,"%1.3e",Glob_Min[current_field_index]);
	glColor3f(color[0],color[1],color[2]);
	glScaled(factor,factor,factor);
	glTranslated(tan(FOV_angle)*cam_posz_0/(2.8*factor)+(cam_posx-cam_posx_0)/factor,
		tan(FOV_angle)*cam_posz_0/(2.55*FOV_aspect_ratio*factor)+(cam_posy-cam_posy_0)/factor,
		(cam_posz-cam_posz_0-0.1)/factor);
	glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
	glPopMatrix();
	
	glPushMatrix();
	sprintf(ToDraw,"%1.3e",Glob_Max[current_field_index]);
	glColor3f(color[0],color[1],color[2]);
	glScaled(factor,factor,factor);
	glTranslated(tan(FOV_angle)*cam_posz_0/(2.8*factor)+(cam_posx-cam_posx_0)/factor,
		tan(FOV_angle)*cam_posz_0/(2.55*FOV_aspect_ratio*factor)+bar_height/factor+(cam_posy-cam_posy_0)/factor,
		(cam_posz-cam_posz_0-0.1)/factor);
	glutStrokeString(GLUT_STROKE_ROMAN,ToDraw);
	glPopMatrix();
	
	glPushMatrix();
	left = tan(FOV_angle)*cam_posz_0/2.1+cam_posx-cam_posx_0;
	bottom = tan(FOV_angle)*cam_posz_0/(2.5*FOV_aspect_ratio)+cam_posy-cam_posy_0;
	dist = cam_posz-cam_posz_0-0.1;
	glTranslated(left,bottom,dist);
	float* Verts = ColorLegendVertices(bar_width,bar_height,quadnum);
	float* Colors = ColorLegendColors(quadnum,&Rainbow_CMap);
	glPolygonMode(GL_FRONT,GL_FILL);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3,GL_FLOAT,0,Verts);
	glColorPointer(3,GL_FLOAT,0,Colors);
	glDrawArrays(GL_QUADS,0,4*(quadnum+2));
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glPopMatrix();
	glLineWidth(old_width);
	free(Verts);
	free(Colors);
	
	if (show_vectors){
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);
	}
}

void draw_selection(){
	if (selection){		
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		int width = glutGet(GLUT_WINDOW_WIDTH);
		
		GLint old = 1;
		glGetIntegerv(GL_LINE_WIDTH,&old);
		
		glLineWidth(2.0);
		glMatrixMode (GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0,width,0,height,-100.,100.);
		glColor3f(0.0, 0.0, 0.0);
		
		glBegin(GL_LINE_STRIP);
		glVertex2i(selection_p1x,selection_p1y);
		glVertex2i(selection_p2x,selection_p1y);
		glVertex2i(selection_p2x,selection_p2y);
		glVertex2i(selection_p1x,selection_p2y);
		glVertex2i(selection_p1x,selection_p1y);
		glEnd();
		
		glPopMatrix();		
		glFlush();
		glLineWidth(old);
	}
}

void draw_segments(){
	if (show_isolines && Segments!=NULL){
		int i;
		
		if (draw_mode==WIRES) glColor3f(1.0,1.0,1.0); else glColor3f(0.8,0.8,0.8);
		
		glBegin(GL_LINES);
		for (i=0;i<Segment_num-1;i++){			
			glVertex3f(Segments[2*i].x,Segments[2*i].y,0.0001);
			glVertex3f(Segments[2*i+1].x,Segments[2*i+1].y,0.0001);		
		}
		glEnd();	
		
		glPopMatrix();		
		glMatrixMode(GL_MODELVIEW);
	}
}

void draw_vector_field(){
	
	const float dz = 0.0001;
	const float linewidth = 1.5;
	const float scale0 = 0.05;
	const float thres = 0.15;
	const float arrow_len = scale0/5.;
	const float arrow_wid = scale0/12.;
	
	int i;
	float x,y,s,X,Y,ox,oy,oldw,nx,ny,tx,ty;
	
	if (!show_vectors || VectorField==NULL) return;
	
	glEnable( GL_POLYGON_SMOOTH );
	glHint( GL_POLYGON_SMOOTH_HINT,GL_NICEST);
	
	glGetFloatv(GL_LINE_WIDTH,&oldw);	
	glLineWidth(linewidth);
	
	if (draw_mode==WIRES) glColor3f(1.0,1.0,1.0); else glColor3f(0.2,0.2,0.2);
	
	int n = glob_mesh.size;
	float scale = scale0/Glob_Max[1];
	
	i = 0;
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		if (frame==DEFORMED){
			x += Displacement[i];
			y += Displacement[n+i];
		}
		s = vec_norm_2D(VectorField[i],VectorField[n+i]);
		if (s*scale>thres*scale0){		
			vec_norm(VectorField[i],VectorField[n+i],&tx,&ty);	
			vec_ortho_norm(tx,ty,&nx,&ny);		
			X = x+scale*VectorField[i];
			Y = y+scale*VectorField[n+i];
			ox = (X-x)/2.;
			oy = (Y-y)/2.;
			x -= ox;
			y -= oy;
			X -= ox;
			Y -= oy;
		
			glBegin(GL_LINES);
			glVertex3f(x,y,dz);
			glVertex3f(X,Y,dz);		
			glEnd();	
		
			glBegin(GL_POLYGON);
			glVertex3f(X,Y,2.*dz);
			glVertex3f(X-arrow_len*tx-arrow_wid*nx,Y-arrow_len*ty-arrow_wid*ny,2.*dz);
			glVertex3f(X-arrow_len*tx+arrow_wid*nx,Y-arrow_len*ty+arrow_wid*ny,2.*dz);
			glEnd();
		}
	}
	
	glDisable(GL_POLYGON_SMOOTH);
	glLineWidth(oldw);
}

void renderMesh(){
	const int DIM = 3;
	const int COL = 4;
	const float lwidth = 1.;
	
	float oldw;
	
	int n = glob_mesh.size;
	int N = elements.size;
	
	glGetFloatv(GL_LINE_WIDTH,&oldw);	
	
	switch(draw_mode){
		case FILL: glPolygonMode(GL_FRONT,GL_FILL);break;
		case WIRES:
			glPolygonMode(GL_FRONT,GL_LINE); 
			glLineWidth(lwidth);
			break;
		case POINTS: glPolygonMode(GL_FRONT,GL_POINT);break;
	}
	
	float* Verts = ToVertexTriangleArray(&No_Height);
	float* Colors = color_map(DataArray,&Rainbow_CMap,ALPHA);
	float* Normals = get_normal_field(&No_Height);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	glVertexPointer(DIM,GL_FLOAT,0,Verts);
	glColorPointer(COL,GL_FLOAT,0,Colors);
	glNormalPointer(GL_FLOAT,0,Normals);
	
	glDrawArrays(GL_TRIANGLES,0,3*N);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	
	free(Verts);
	free(Normals);
	free(Colors);
	
	glLineWidth(oldw);
}

void renderScene(){
	
	init_scene();
	init_lighting();
	renderMesh();
	draw_vector_field();
	draw_segments();
	draw_legends();
	draw_selection();
	
	glutSwapBuffers();
}

void make_movie(int start,int end,int field_index){
	
	const char* iformat = "png";
	const char* oformat = "mpg";
	
	int i,p;
	char Name[512];
	char DirName[512];
	char Input_files[512];
	char Output_file[512];
	char Command[1024];
	char* Number;
	
	// create temp dir
	sprintf(DirName,"%s/%s/temp",Home,DataFileName);
	mkdir(DirName,0777);
	
	// render images
	printf("rendering movie ... ");
	fflush(stdout);
	for (i=start;i<=end;i += step_size){
		current_timestep = i;
		load_data_at_time(i,field_index);
		renderScene();		
		Number = FillZero(i-start,end-start);
		sprintf(Name,"%s/%s/temp/img_%s.png",Home,DataFileName,Number);
		if (i>start) save_screen(Name);
		free(Number);
		p = (int)round((double)100.*(i+1-start)/(end+1-start));
		printf("\rrendering movie ... %d %%",p);
		fflush(stdout);
	}	
	
	printf("\rrendering movie ... done \n");
	
	// start mencoder
	sprintf(Input_files,"%s/%s/temp/img_*.%s",Home,DataFileName,iformat);
	sprintf(Output_file,"%s/%s/movie%d.%s",Home,DataFileName,field_index,oformat);
	sprintf(Command,"mencoder -ovc lavc -mf fps=%d:type=%s mf://%s -o %s > mencoder_log",
		frames_per_sec,iformat,Input_files,Output_file);
	printf("execute: %s\n",Command);
	system(Command);
	
	// remove temp
	sprintf(Command,"rm -r %s/%s/temp",Home,DataFileName);
	system(Command);
}

void mouse_pushed(int button,int state,int x, int y){
	if (selection && button==GLUT_LEFT_BUTTON && state== GLUT_UP){
		//double px,py;
		
		//double pz = 0;
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		//MouseToWorld(x,height-y,&px,&py,pz);
		//current_mouse_pos.x = px;
		//current_mouse_pos.y = py;
		if (!selection_frame){
			selection_p1x = x;
			selection_p1y = height-y;
			selection_frame = 1;
		}
		else{
			selection_p2x = x;
			selection_p2y = height-y;
			selection_frame = 0;
		}		
	}
}

void mouse_moved(int x, int y){
	if (selection && selection_frame){
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		selection_p2x = x;
		selection_p2y = height-y;
	}
}

void special_keystruck(int key, int x, int y){
	static double G[4];
	
	char Name[512];
	int update = 0;
	int mod = glutGetModifiers();
	
	switch(key){
		case GLUT_KEY_PAGE_UP:
			current_timestep += step_size;
			//current_timestep++;
			if (time_step_range!=NULL){
				if (current_timestep>time_step_range->j) current_timestep--;
			}
			update = 1;
			break;
		case GLUT_KEY_PAGE_DOWN:
		current_timestep -= step_size;
			//current_timestep--;
			if (time_step_range!=NULL){
				if (current_timestep<time_step_range->i) current_timestep++;
			}			
			update = 1;
			break;
		case GLUT_KEY_LEFT:
			if (mod==GLUT_ACTIVE_CTRL) glob_angle_z -= M_PI*0.1;
			else cam_posx -= 0.1*cam_posz;
			break;
		case GLUT_KEY_RIGHT: 
			if (mod==GLUT_ACTIVE_CTRL) glob_angle_z += M_PI*0.1;
			else cam_posx += 0.1*cam_posz;
			break;		
		case GLUT_KEY_UP: cam_posy += 0.1*cam_posz;break;
		case GLUT_KEY_DOWN: cam_posy -= 0.1*cam_posz;break;
		case GLUT_KEY_END: 
			if (selection) selection = 0; else selection = 1;
			update = 1;
			break;
		case GLUT_KEY_INSERT: 
			if (mod==GLUT_ACTIVE_CTRL){
				//make_movie(0,timestep_num-1,current_field_index); 
				make_movie(0,current_timestep,current_field_index); 
				save_selection();
				exit(0);
			}
			else{
				sprintf(Name,"%s/%s/field%dindex%d.png",Home,DataFileName,current_field_index,current_timestep);
				printf("save image, field index: %d time index: %d\n",current_field_index,current_timestep);
				save_screen(Name);
				save_selection();
			}
			break;
		case GLUT_KEY_F1: draw_mode = FILL;break;
		case GLUT_KEY_F2: draw_mode = WIRES;break;	
		case GLUT_KEY_F3: draw_mode = POINTS;break;		
		case GLUT_KEY_F4: show_damage = !show_damage;break;		
		case GLUT_KEY_F5: selection_deactivated = !selection_deactivated;break;		
		case GLUT_KEY_F6: 
			show_isolines = !show_isolines;
			update = 1;
			break;	
		case GLUT_KEY_F7: 
			get_structure_matrix(G,DataArray);
			printf("\nstructure matrix:\txx=%f\txy=%f\tyy=%f\n\n",G[0],G[2],G[3]);
			fflush(stdout);
			break;
	}
	if (current_timestep<0) current_timestep = 0;
	if (current_timestep>=timestep_num) current_timestep = timestep_num-1;
	if (update) load_data_at_time(current_timestep,current_field_index);
}

void keystruck(unsigned char key, int x, int y){
	const double a = 0.67;
	
	double px,py;
	double pz = 0;
	
	int height = glutGet(GLUT_WINDOW_HEIGHT);
	int update = 0;
	int old = current_field_index;
	int mod = glutGetModifiers();
	switch(key){
		case '+': cam_posz *= a;break;
		case '-': cam_posz /= a;break;
		case ' ': frame = (frame==REFERENCE ? DEFORMED : REFERENCE);break;
		case '1': 		
			current_field_index = 0;
			update = 1;
			break;
		case '2': 
			current_field_index = 1;
			update = 1;
			break;
		case '3':
			current_field_index = 2;
			update = 1;
			break;
		case '4':
			current_field_index = 3;
			update = 1;
			break;
		case '5':
			current_field_index = 4;
			update = 1;
			break;
		case '6':
			current_field_index = 5;
			update = 1;
			break;
		case '7':
			current_field_index = 6;
			update = 1;
			break;
		case '8':
			current_field_index = 7;
			update = 1;
			break;	
		case '9': 
			current_field_index = 8;
			update = 1;
			break;
		case '0': 
			current_field_index = 9;
			update = 1;
			break;
		case 'q':
			current_field_index = 10;
			update = 1;
			break;
		case 'w':
			current_field_index = 11;
			update = 1;
			break;
		case 'e':
			current_field_index = 12;
			update = 1;
			break;		
		case 'r':
			current_field_index = 13;
			update = 1;
			break;	
		case KEY_TAB: 
			MouseToWorld(x,height-y,&px,&py,pz);
			current_mouse_pos.x = px;
			current_mouse_pos.y = py;
			current_node_index = Nearest_index(&current_mouse_pos);
			break;
		case 'l':
			local_minmax(DataArray,current_field_index);
			update = 1;
			break;
	}
	if (update){
		if (current_field_index>=data_field_num) current_field_index = old;
		else load_data_at_time(current_timestep,current_field_index);
	}
}

void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	
	if (h == 0) h = 1;
	float ratio = (float)w*1.0/h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45.,ratio,1,1000);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
}

void init(){
	const int BUFF_SIZE = 512;
	
	// init camera position
	cam_posz = cam_posz_0;
	cam_posx = cam_posx_0;
	cam_posy = cam_posy_0;
	
	// init mouse position
	current_mouse_pos.x = 0;
	current_mouse_pos.y = 0;
	
	// load data.bin.info file
	int i;
	char Fullname[512];
	sprintf(Fullname,"%s/%s/%s.info",Home,DataFileName,Bin_name);
	FILE* Names = fopen(Fullname,"r");
	if (Names!=NULL){
		char Buffer[BUFF_SIZE];
		fgets(Buffer,BUFF_SIZE,Names);
		int num = atoi(Buffer);
		if (num!=data_field_num){
			printf("Warning: inconsistent info file found: %d required: %d-> skip\n",num,data_field_num);
			fclose(Names);
			return;
		}
		else{
			Field_Names = (char**)malloc(BUFF_SIZE*sizeof(char*));
			for (i=0;i<data_field_num;i++){
				Field_Names[i] = (char*)malloc(BUFF_SIZE*sizeof(char));
				fgets(Buffer,BUFF_SIZE,Names);
				sprintf(Field_Names[i],"%s",Buffer);					
			}
			fclose(Names);
		}
	}
	else printf("Warning could not open file %s -> skip\n",Fullname);
}

void process_args(int argc, char *argv[], int start){
	int i,j,len,slen;
	char** Parts;
	for (i=start;i<argc;i++){
		Parts = split(argv[i],"=",&len);
		//if (len!=2) printf("Warning: something's wrong with arg #%d !\n",i);
		
		if (strcmp(Parts[0],"-time")==0){
			if (strcmp(Parts[1],"last")==0) current_timestep = LAST_TIME_STEP;
			else current_timestep = atoi(Parts[1]);
			
		}
		else if (strcmp(Parts[0],"-field")==0) current_field_index = atoi(Parts[1]);
		else if (strcmp(Parts[0],"-dir")==0) sprintf(Home,"%s",Parts[1]);
		else if (strcmp(Parts[0],"-movie")==0) Program_Mode = MAKEMOVIE;
		else if (strcmp(Parts[0],"-deform")==0) frame = DEFORMED;
		else if (strcmp(Parts[0],"-interval")==0) step_size = atoi(Parts[1]);
		else if (strcmp(Parts[0],"-fps")==0) frames_per_sec = atoi(Parts[1]);
		else if (strcmp(Parts[0],"-range")==0){											// exmaple range format: -range=1*2.0*3.0
			range_mode = MANUAL_RANGE;
			slen = 0;
			char** SubParts = split(Parts[1],"*",&slen);
			
			if (slen==3){
				man_range_index = atoi(SubParts[0]);
				if (strcmp(SubParts[1],"min")!=0) man_range_min = atof(SubParts[1]);else man_range_min = NAN;
				if (strcmp(SubParts[2],"max")!=0) man_range_max = atof(SubParts[2]);else man_range_max = NAN;
				//printf("p1=%s p2=%s p3=%s\n",SubParts[0],SubParts[1],SubParts[2]);
			}
			for (j=0;j<slen;j++) free(SubParts[j]);
			free(SubParts);
		}
		else if (strcmp(Parts[0],"-irange")==0){
			slen = 0;
			char** SubParts = split(Parts[1],"*",&slen);
			if (slen==2){
				time_step_range = (index2D*)malloc(sizeof(index2D));
				time_step_range->i = atoi(SubParts[0]);
				time_step_range->j = atoi(SubParts[1]);
			}
			for (j=0;j<slen;j++) free(SubParts[j]);
			free(SubParts);
		}
		else printf("Warning: argument '%s' could not be processed\n",argv[i]);
		
		for (j=0;j<len;j++) free(Parts[j]);
		free(Parts);
	}
}

void simple_view(int argc,char** argv){					// example: FEMviewer simple /Home/damage/radszuwe/Daten/poly_0 /Home/damage/radszuwe/Daten/newatr.1 0 1

	Mesh_name = (char*)malloc(512*sizeof(char));
	
	Glob_Min = zero_vector(1);
	Glob_Max = zero_vector(1);
	Times = zero_vector(1);
	
	Mesh_number = 1;
	timestep_num = 1;
	data_field_num = 1;
	
	sprintf(DataFileName,"%s",argv[2]);
	sprintf(Mesh_name,"%s",argv[3]);
	if (argc==6){
		Glob_Min[0] = atof(argv[4]);
		Glob_Max[0] = atof(argv[5]);
	}	
	
	load_mesh(Mesh_name,&glob_mesh,&elements,QUIET);
	
	Mesh_sizes = zero_int_list(Mesh_number);
	Mesh_sizes[0] = glob_mesh.size;
	int n = Mesh_sizes[0];
	Displacement = zero_vector(2*n);
	Damage = generate_vector(n,1.);
	
	if (strcmp(argv[1],"simple")==0) DataArray = load_data(DataFileName,n);
	else if (strcmp(argv[1],"vsimple")==0){
		DataArray = load_data(DataFileName,3*n);
		VectorField = &(DataArray[n]);
		Glob_Min = (double*)realloc(Glob_Min,2*sizeof(double));
		Glob_Max = (double*)realloc(Glob_Max,2*sizeof(double));
		minmax_abs_2D(VectorField,n,&(Glob_Min[1]),&(Glob_Max[1]));
		show_vectors = 1;
	} 
	
	/*int i;
	double offset = 0;
	for (i=0;i<n;i++){
		double y = glob_mesh.Points[i].y;
		DataArray[i] += y;
		if (y==1) offset = DataArray[i];
	}
	for (i=0;i<n;i++) DataArray[i] -= offset;*/
	
	if (strcmp(argv[4],"auto")==0) minmax(DataArray,n,Glob_Min,Glob_Max);
	
}

int main(int argc, char *argv[]){
	
	char Name[512];
	char* home = getenv("HOME");
	sprintf(Home,"%s/Daten",home);
	
	if (strcmp(argv[1],"simple")==0 || strcmp(argv[1],"vsimple")==0){
		Program_Mode = SIMPLEVIEW;
		simple_view(argc,argv);
	}
	else if (strcmp(argv[1],"-h")==0){
		printf("#1:\tsimple view mode to load a mesh and a data text file independently\n");
		printf("\t\t#2: data file (with path)\n");
		printf("\t\t#3: mesh name (with path)\n");
		printf("\t\t#4: minimum value\n");
		printf("\t\t#5: miximum value\nor\n");		
		printf("#1:\tfilename of data file (normal mode)\n");
		printf("\t\t#2-...: options, \"-OPTNAME=<value>\"\n");
		printf("\t\t-dir: Path  to data file if it is not in standard dir \"/Home/damage/radszuwe/Daten\"\n");
		printf("\t\t-time: time index of step to display\n");
		printf("\t\t-field: field index to display at first (type last for \"last\" timestep)\n");
		printf("\t\t-range: value range for the specified field, format: <field>*<min>*<max> (use \"min\" or \"max\" for minimum and maximum respectively)\n");
		printf("\t\t-irange: load only a certain range of timesteps is much faster), format: <min index>*<max index> (\"time\" must be in the specifield interval)\n");
		printf("\t\t-deform: switch to draw deformed mesh configuration\n");
		printf("\t\t-movie: make movie imidiately\n");
		exit(0);
	}
	else{
		sprintf(DataFileName,"%s",argv[1]);
		process_args(argc,argv,2);
	
		// load mesh info
		get_mesh_name(&Mesh_name);
		load_mesh_sizes(&Mesh_sizes,&Mesh_number);
		Null_mesh(&glob_mesh);
		elements.Elements = NULL;
	
		// load data (binary)
		sprintf(Name,"%s/%s/%s",Home,DataFileName,Bin_name);
		total_minmax(Name,&Glob_Min,&Glob_Max,&Times,&timestep_num,&data_field_num);
		
		// load selection		
	}
	
	init();
	
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(2048,1024);
    glutInitWindowPosition(100,100);
    glutCreateWindow("FEMviewer");
    if (!load_selection()){
		selection_p2x = glutGet(GLUT_WINDOW_WIDTH);
		selection_p2y = glutGet(GLUT_WINDOW_HEIGHT);
	}
	
	// register callbacks
	glutDisplayFunc(&renderScene);
	glutIdleFunc(&renderScene);
	glutReshapeFunc(&changeSize);
	glutKeyboardFunc(&keystruck);
	glutSpecialFunc(&special_keystruck);
	glutMouseFunc(&mouse_pushed);
	glutPassiveMotionFunc(&mouse_moved);
	
	// OpenGL init
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	//glEnable(GL_LIGHTING);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
	
	if (show_vectors){
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);		
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	}
	
	//load_data_at_time(current_timestep,current_field_index);
	
	isoline* Iso;
	
	// enter GLUT event processing cycle
	switch(Program_Mode){
		case NORMAL:
			load_data_at_time(current_timestep,current_field_index);					
			glutMainLoop();
			break;
		case MAKEMOVIE:
			load_data_at_time(current_timestep,current_field_index);
			make_movie(0,timestep_num-1,current_field_index); 
			break;
		case SIMPLEVIEW: 
			glutMainLoop();
			break;
	}
	
	free(DataArray);
	free(Damage);
	free(Displacement);
	return 0;
}
