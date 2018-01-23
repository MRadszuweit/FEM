#ifndef CUDA_LINEAR_ALGEBRA_H
#define CUDA_LINEAR_ALGEBRA_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include "linear_algebra.h"

#define DEFAULT 0
#define STORE_INV_DIAG 1

// Datentypen ///////////////////////////////

typedef struct CSR_MATRIX{
	int total_size;
	int row_num;
	float* Val;
	int* Ind;
	int* Start_index;
	float* Inv_diag;    // vector of inverse diagonal
	
}csr_matrix;

// Funktionen /////////////////////////////////////

cudaError_t cpy_df_to_device(float* _X,double* X,int n);
cudaError_t cpy_fd_to_host(double* X,float* _X,int n);
void device_print_vector(float* _X,int n);
void convert_sparse_to_csr(csr_matrix* S,sparse_matrix* A, int store_inv_diag);
void sparse_to_csr_device(csr_matrix* _S, sparse_matrix* A, int store_inv_diag);
void free_csr_device(csr_matrix* _S);

__global__ void cuda_vector_add(float* X,float* Y,float factor,int n);
__global__ void cuda_vector_mult_add(float* X,float* Y,float factor,int n);
__global__ void cuda_pseudo_mult(float* _X,float* _Y,int n);

void print_dev_prop();
float* cuda_zero_vector(int n);
float* cuda_get_vector_on_device(double* X,int n);
float cuda_scalar(float* _X, float* _Y,int n);
float cuda_euklid_norm(float* _X,int n);
void cuda_gpu_settings(int deviceID);
void cuda_sparse_on_device(sparse_matrix* A,int max_bandwidth,int** _Indices,float** _Values);
//void cuda_jacobi(double* X,double* b,int* _Indices,double* _Values,int n,int max_bandwidth,int iter);
void cuda_csr_jacobi(csr_matrix* _S,float* _X,float* _b,int iter);

#endif
