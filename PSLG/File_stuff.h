 /* ilb: file_stuff */

#ifndef FILE_STUFF_H
#define FILE_STUFF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Funktionen /////////////////////////////////////

FILE* open_file(char* dir,char* name,char* mode);
void append_data(FILE* file,double* data,int size,int var_num,double time);

#endif
