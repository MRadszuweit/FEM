 /* ilb: file_stuff */

#ifndef FILE_STUFF_H
#define FILE_STUFF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Funktionen /////////////////////////////////////

int readline(FILE* file,char* buffer);
int readLine(FILE* file,char* buffer,int buff_size);
//char** split(char* string,char a,int* n);
char** split(char* String,char* delimiter,int* Number);
FILE* open_file(char* dir,char* name,char* mode);
FILE* Open_file(char* filename,char* mode);
void append_data(FILE* file,double* data,int size,int var_num,double time);
void write_lists(char* path,char* name,int** Lists,int number,int size);
int** read_lists(char* path,char* name,int* Number,int* Size);
#endif
