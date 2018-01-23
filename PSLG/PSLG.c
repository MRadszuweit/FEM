/* creates PSLG-geometry for Triangle */

#include <math.h>
#include <stdio.h>
#include "geometry2D.h"
#include "File_stuff.h"


// globale Variablen ///////////////////

static FILE* File;

// Code ////////////////////////////////

void init(int n){
	char* name = "disc.poly";
	char* dir = "/home/radszuweit/Daten/meshes2D";
	File = open_file(dir,name,"w");
}

double create_disc_geometry(int n,double r,point2D* data){
	int i;
	double phi;
	for (i=0;i<n;i++){
		phi = (double)2*M_PI*i/n;
		data[i].x = r*sin(phi);
		data[i].y = r*cos(phi);
	}
	double d = dist(&data[0],&data[1]);
	printf("area magnitude: %f\n",d*d/2);
	return d*d/2;
}

double create_bidomain_disc(int n1,int n2,double r1,double r2,point2D* data){
	int i;
	double phi;
	for (i=0;i<n1;i++){
		phi = (double)2*M_PI*i/n1;
		data[i].x = r1*sin(phi);
		data[i].y = r1*cos(phi);
	}
	for (i=0;i<n2;i++){
		phi = (double)2*M_PI*i/n2;
		data[i+n1].x = r2*sin(phi);
		data[i+n1].y = r2*cos(phi);
	}
	double d = dist(&data[0],&data[1]);
	printf("area magnitude: %f\n",d*d/2);
	return d*d/3;
}

double create_pseudo_1D(int n,int m,double L,point2D* data){
	int i;
	double dx = (double)L/(n-1);
	for (i=0;i<n;i++){
		data[i].x = (double)i*dx;
		data[i+m+n].x = L-data[i].x;
		data[i].y = (double)(m+1)*dx;
		data[i+m+n].y = 0;
	}
	for (i=0;i<m;i++){
		data[n+i].x = L;
		data[2*n+m+i].x = 0;
		data[n+i].y = (double)(m-i)*dx;
		data[2*n+m+i].y = (i+1)*dx;
	}
	printf("area magnitude: %f\n",sqrt(3.)*dx*dx/4.);
	return sqrt(3.)*dx*dx/4.;
}

double create_partitioned_1D(int n,int m,double L,double x1,double x2,point2D* data){
	int i;
	double dx = (double)L/(n-1);
	for (i=0;i<n;i++){
		data[i].x = (double)i*dx;
		data[i+m+n].x = L-data[i].x;
		data[i].y = (double)(m+1)*dx;
		data[i+m+n].y = 0;
	}
	for (i=0;i<m;i++){
		data[n+i].x = L;
		data[2*n+m+i].x = 0;
		data[n+i].y = (double)(m-i)*dx;
		data[2*n+m+i].y = (double)(i+1)*dx;
		data[2*n+2*m+i].x = x1;
		data[2*n+3*m+i].x = x2;
		data[2*n+2*m+i].y = (double)(m-i)*dx;
		data[2*n+3*m+i].y = (double)(i+1)*dx;
	}
	double a = 0.5;
	data[2*n+4*m].x = a*dx;
	data[2*n+4*m].y = a*dx;
	data[2*n+4*m+1].x = L-a*dx;
	data[2*n+4*m+1].y = a*dx;
	data[2*n+4*m+2].x = a*dx;
	data[2*n+4*m+2].y = (double)(m+1)*dx-a*dx;
	data[2*n+4*m+3].x = L-a*dx;
	data[2*n+4*m+3].y = (double)(m+1)*dx-a*dx;
	printf("area magnitude: %f\n",sqrt(3.)*dx*dx/2.);
	return sqrt(3.)*dx*dx/2.;
}

void create_poly_file(point2D* data,int total_size,int* attr_sizes,point2D* attr_points,int attr_num){
	int i,k;
	point2D p;
	int n = 0;
	for (k=0;k<attr_num;k++) n += attr_sizes[k]; 
	fprintf(File,"%d %d %d %d\n",total_size,2,attr_num,0);
	for (i=0;i<total_size;i++){
		p = data[i];
		fprintf(File,"%d %f %f\n",i,p.x,p.y);
	}
	fprintf(File,"%d %d\n",n,0);
	int start = 0;
	for (k=0;k<attr_num;k++){
		fprintf(File,"%d %d %d\n",start,start+attr_sizes[k]-1,start);
		for (i=start+1;i<start+attr_sizes[k];i++){
			fprintf(File,"%d %d %d\n",i,i-1,i);
		}
		start += attr_sizes[k];
	}
	fprintf(File,"%d\n",0);
	if (attr_num>1){
		fprintf(File,"%d\n",attr_num);
		for (i=0;i<attr_num;i++){
			fprintf(File,"%d %f %f %d %d\n",i,attr_points[i].x,attr_points[i].y,i,-1);
		}
	}
}

int main(int argc, char* argv[]){
	int i,j,k,num;
	int n = argc-2;
	int m = argc-1;
	double r,a,R;
	point2D* Data;
	char* s = (char*)malloc(256*sizeof(char));
	char* name = (char*)malloc(256*sizeof(char));
	char* name1 = (char*)malloc(256*sizeof(char));
	char* name2 = (char*)malloc(256*sizeof(char));
	char* dir = "/users/radszuweit/Daten";
	r = 1;
	R = 0.7;
	num = 2;
	//printf("%s\n",argv[m]);
	for (i=n;i>0;i--){
		j = atoi(argv[i]);
		sprintf(name,"%s.poly",argv[m]);
		File = open_file(dir,name,"w");
		
		int h = 7;									//für 1D
		double J = (double)R/r*j; // anzahl innen:
		int* Sizes = (int*)malloc(num*sizeof(int));
		point2D* Markers = (point2D*)malloc(num*sizeof(point2D));
		/*Markers[0].x = 0.1;  // partitioned 1D
		Markers[0].y = 0.1;
		Sizes[0] = j;
		Markers[1].x = 1.0;
		Markers[1].y = 0.1;
		Sizes[1] = h;
		Markers[2].x = 1.9;
		Markers[2].y = 0.1;
		Sizes[2] = h;
		int N = 4;									// number of extra points */
		Markers[0].x = 0.0;		// bidisc
		Markers[0].y = 0.0;
		Sizes[0] = j;
		Markers[1].x = 0.9;
		Markers[1].y = 0.0;
		Sizes[1] = J;
		int N = 0;
		for (k=0;k<num;k++) N += Sizes[k];
		
		Data = (point2D*)malloc(N*sizeof(point2D));
		a = create_bidomain_disc(j,(int)J,r,R,Data);
		//a = create_disk_geometry(j,r,Data);
		//a = create_partitioned_1D((j-2*h)/2,h,2.*r,0.3*r,1.7*r,Data);
		//a = create_pseudo_1D((j-2*h)/2,h,2.*r,Data);
		create_poly_file(Data,N,Sizes,Markers,num);
		free(Data);
		fclose(File);
		//sprintf(s,"triangle -p -D -v -q30 -a%f %s/%s",a,dir,name);
		sprintf(s,"triangle -p -D -F -v -q30 -A -a%.10f %s/%s",a,dir,name);  // für bidomain disc
		printf("%s\n",s);
		system(s);
		if (i>1){
			sprintf(name1,"%s.1.poly",argv[m]);
			sprintf(name2,"%s.%d.poly",argv[m],i);
			sprintf(s,"mv %s/%s %s/%s ",dir,name1,dir,name2);
			system(s);
			sprintf(name1,"%s.1.ele",argv[m]);
			sprintf(name2,"%s.%d.ele",argv[m],i);
			sprintf(s,"mv %s/%s %s/%s ",dir,name1,dir,name2);
			system(s);
			sprintf(name1,"%s.1.node",argv[m]);
			sprintf(name2,"%s.%d.node",argv[m],i);
			sprintf(s,"mv %s/%s %s/%s ",dir,name1,dir,name2);
			system(s);
		}
	}
	sprintf(s,"rm %s.poly",argv[m]);
	//printf("%s\n",s);
	system(s);
	printf("Konstruktion von %d Gittern erfolgreich\n\n",n);
	return 0;
}
