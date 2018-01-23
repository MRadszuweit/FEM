
#include "File_stuff.h"

// global Variablen //////////////////////////////////////
 
// Code ///////////////////////////////////////////

FILE* open_file(char* dir,char* name,char* mode){
	char filename[512];
	sprintf(filename,"%s/%s",dir,name);
	FILE* file = fopen(filename,mode);
	if (file==NULL){
		printf("Datei %s konnte nicht angelegt werden\n",filename);
		exit(0);
	}
	return file;
}

void append_data(FILE* file,double* data,int size,int var_num,double time){
	int i,j,k,len;
	char buffer[512];
	char dat[64];
	fprintf(file,"t = %lf\n", time);
	for (i=0;i<size;i++){
		len = 0;
		strcpy(buffer,"\0");
		for (j=0;j<var_num;j++){
			k = size*j+i;
			sprintf(dat,"%lf\t",data[k]);
			strcat(buffer,dat);
			len += strlen(dat);
		}
		buffer[len-1] = '\0';
		fprintf(file,"%s\n",buffer);
	}
	fflush(file);
}



	/*char fn[512]; 
	
	sprintf(fn, "basenem%d%s.txt", 10, "hallo_leute");
	
	FILE *f = fopen(fn, "w");
	
	for (i = 0; i < ; i++)
	{
		fprintf(f, "%lf\t%lf\n", 1.2, 1.5);
	}
	
	fclose(f);*/
	




