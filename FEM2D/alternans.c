#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "linear_algebra.h"

// defines

//////// global vars ///////////////////////////////////////////

int N;  // time steps
int M;  //space steps
int T_size;  // # of Ts
int DI_size;  // # of DIs
int APD_size;  // # of APDs

double L; // spatial length
double total_time; // sim time
double x;  // space
double t;  // time
double dx;  // space step
double dt;  // time step

double* v;  // normalized voltage
double* F;  // deformation gradient

double** T;  // period
double** DI;  // diastolic interval
double** APD;  // action potential duration

//////// programm //////////////////////////////////////////////

// initialization //

void init(int load){
	
	// discretization
	
	dt = 0.01;
	total_time = 10.;
	N = (int)ceil(total_time/dt);
	
	dx = 0.1;
	L = 8.0;
	M = (int)ceil(L/dx);
	
	v = zero_vector(M);
	F = zero_vector(M);
	
	// parameters
	
	
}










int main(int argc, char* argv[]){
	
	
	
	
	
	return 0;
}

