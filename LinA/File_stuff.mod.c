#ifdef _POMP
#  undef _POMP
#endif
#define _POMP 200110

#include "File_stuff.c.opari.inc"
#line 1 "File_stuff.c"

#include "File_stuff.h"

// global Variablen //////////////////////////////////////
 
// Code ///////////////////////////////////////////

FILE* open_file(char* dir,char* name,char* mode){
	FILE* file;
	char filename[512];
	if (name[0]=='/') sprintf(filename,"%s",name); else sprintf(filename,"%s/%s",dir,name);
	file = fopen(filename,mode);
	if (strcmp(mode,"r")==0){
		if (file==NULL) printf("Datei %s existiert nicht\n",filename);
		return file;
	}
	else{
		if (file==NULL){
			printf("Datei %s konnte nicht angelegt/gelesen werden\n",filename);
			exit(0);
		}
	}
	return file;
}

FILE* Open_file(char* filename,char* mode){
	FILE* file = fopen(filename,mode);
	if (file==NULL && strcmp(mode,"r")==0){
		printf("Datei %s existiert nicht\n",filename);
	}
	return file;
}

void append_data(FILE* file,double* data,int size,int var_num,double time){
	int i,j,k,len;
	fprintf(file,"t = %f\n", time);
	for (i=0;i<size;i++){
		len = 0;
		char buffer[2048] = "";
		for (j=0;j<var_num;j++){
			char dat[256] = "";
			k = size*j+i;
			sprintf(dat,"%lf ",data[k]);
			strcat(buffer,dat);
		}
		fprintf(file,"%s\n",buffer);
	}
	fflush(file);
}

/*char** split(char* string,char a,int* n){
	int i,j,counter;
	int len = strlen(string);
	j = 0;
	counter = 0;
	char** Parts = (char**)malloc(sizeof(char*));
	Parts[counter] = (char*)malloc(sizeof(char));
	for (i=0;i<len;i++){
		if (string[i]==a){
			Parts[counter][j] = '\0';
			j = 0;
			counter++;
			Parts = (char**)realloc(Parts,(counter+1)*sizeof(char*));
			Parts[counter] = (char*)malloc(sizeof(char));
			if (i==len-1){
				Parts = (char**)realloc(Parts,counter*sizeof(char*));
				counter--;
				}
		}
		else{
			Parts[counter][j] = string[i];
			j++;
			Parts[counter] = (char*)realloc(Parts[counter],(j+1)*sizeof(char));
		}
	}
	Parts[counter][j] = '\0';
	counter++;
	len = counter;
	i = 0;
	while(i<len){
		if (strlen(Parts[i])==0){
			free(Parts[i]);
			for (j=i+1;j<len;j++){Parts[j-1] = Parts[j];}
			len--;
			Parts = (char**)realloc(Parts,len*sizeof(char*));
		}
		else{i++;}
	}
	*n = len;
	return Parts;
}*/

// 
// name: split
// @param
// @return
char** split(char* String,char* delimiter,int* Number){
	const int buff_size = 2048;
	(*Number) = 0;
	if (String!=NULL){
		//char* string = strdup(String);
		char string[buff_size];
		strncpy(string, String,buff_size);
		char** Res = NULL;
		char* buffer = strtok(string,delimiter);
			while (buffer){
			(*Number)++;
			Res = (char**)realloc(Res,(*Number)*sizeof(char*));
			Res[*Number-1] 	= (char*) malloc(256*sizeof(char));
			//Res[*Number-1] = strdup(buffer);
			strncpy(Res[*Number-1],buffer, 256);
			buffer = strtok(NULL,delimiter);
		}
		//free(string);
		return Res;
	}
	else return NULL;
}
//// 
//// name: split
//// @param
//// @return
//char** split(const char* String,const char* delimiter,int* Number){
	//(*Number) = 0;
	//printf(">>> %s '%s'\n",String, delimiter);
	//if (String!=NULL){
		//char* string = strdup(String);
		//char** Res = NULL;
		//char* buffer = strtok(string,delimiter);
			//while (buffer){
			//(*Number)++;
			//Res = (char**)realloc(Res,(*Number)*sizeof(char*));
			//Res[*Number-1] = strdup(buffer);
			//buffer = strtok(NULL,delimiter);
		//}
		//free(string);
		//exit(0);
		//return Res;
	//}
	//else return NULL;
//}

/*char* readline(FILE* file){
	char* buffer = NULL;
	buffer = (char*)malloc(512*sizeof(char));
	fgets(buffer,256,file);
	if (buffer[strlen(buffer)-1]=='\n'){buffer[strlen(buffer)-1] = '\0';}
	return buffer;
}*/

int readline(FILE* file,char* buffer){
	char c;
	int n = 512;
	int i = 0;
	do{
		//printf("file pointer: %p\n",file);
		c = fgetc(file);
		if (c!='\n' && c!=EOF){
			buffer[i] = c;
			i++;
			if (i==n-1){
				printf("Zeile zu lang: %s\n",buffer);
				return -1;
			}
		}
	}while(c!='\n' && c!=EOF);
	buffer[i] = '\0';
	if (c==EOF) return 0; else return 1;
}

int readLine(FILE* file,char* buffer,int buff_size){
	char c;
	int i = 0;
	do{
		//printf("file pointer: %p\n",file);
		c = fgetc(file);
		if (c!='\n' && c!=EOF){
			buffer[i] = c;
			i++;
			if (i==buff_size-1){
				printf("Zeile zu lang: %s\n",buffer);
				return -1;
			}
		}
	}while(c!='\n' && c!=EOF);
	buffer[i] = '\0';
	if (c==EOF) return 0; else return 1;
}

void write_lists(char* path,char* name,int** Lists,int number,int size){
	int i,j;
	//size_t len;
	char part[512];
	char filename[512];
	//char* line = (char*)malloc(number*sizeof(char)*256);
	char line[512];
	sprintf(filename,"%s/%s",path,name);
	FILE* file = fopen(filename,"w");
	for (i=0;i<size;i++){
		sprintf(line,"%d",Lists[0][i]);
		for (j=1;j<number;j++){
			sprintf(part," %d",Lists[j][i]);
			strcat(line,part);
		}
		//len = strlen(line);
		//line[len] = '\0';
		fprintf(file,"%s\n",line);
	}
	fclose(file);
	//free(line);
}

int** read_lists(char* path,char* name,int* Number,int* Size){						//Listen müssen gleiche Länge haben !
	int i;
	int** Res = NULL;
	char** Parts;
	char filename[512];
	char line[64000];
	sprintf(filename,"%s/%s",path,name);
	FILE* file = fopen(filename,"r");
	*Size = 0;
	while(readline(file,line)){
		Parts = split(line," ",Number);
		if ((*Number)<=0){
			break;
		}
		if ((*Size)==0){
			Res = (int**)malloc((*Number)*sizeof(int*));
			for (i=0;i<(*Number);i++) Res[i] = NULL;
		}
		(*Size)++;
		for (i=0;i<(*Number);i++){
			Res[i] = (int*)realloc(Res[i],(*Size)*sizeof(int));
			Res[i][(*Size)-1] = atoi(Parts[i]);
			free(Parts[i]);
		}
		free(Parts);
	}
	fclose(file);
	return Res;
}
