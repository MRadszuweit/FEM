#include "cudaLinA.h"

// global variables 

int blocks = 1;
int threads = 256;

cusparseStatus_t cL_status;
cusparseHandle_t cL_handle = 0;
cusparseMatDescr_t cL_descra = 0;

//double CG_eps = 1.0e-10;
//int CG_max_iter = 100;
//static void (*cuda_CG_Precon)(double* _Z,double* R_);

// functions ////////////////////////////////////////////////////////////////////////////////////////

void device_print_vector(float* _X,int n){
	cudaError_t status;
	double* X = (double*)malloc(n*sizeof(double));
	status = cpy_fd_to_host(X,_X,n);
	//cudaMemcpy(X,_X,n*sizeof(double),cudaMemcpyDeviceToHost);
	if (status!=cudaSuccess) printf("failed\n");
	else print_vector(X,n);
	free(X);
}

void convert_sparse_to_csr(csr_matrix* S,sparse_matrix* A, int store_inv_diag){
	int i,j,offset;
	int n = A->size;
	S->Start_index = (int*)malloc((n+1)*sizeof(int));
	S->row_num = n;
	S->total_size = 0;
	S->Start_index[0] = 0;
	for (i=0;i<n;i++){
		S->total_size += A->Len[i];
		S->Start_index[i+1] = S->total_size;
	}
	S->Ind = (int*)malloc(S->total_size*sizeof(int));
	S->Val = (float*)malloc(S->total_size*sizeof(float));
	if (store_inv_diag) S->Inv_diag = (float*)malloc(n*sizeof(float)); else S->Inv_diag = NULL;
	for (i=0;i<n;i++){
		offset = S->Start_index[i];
		for (j=0;j<A->Len[i];j++){
			S->Ind[offset+j] = A->Indices[i][j];
			S->Val[offset+j] = (float)A->Values[i][j];
			if (store_inv_diag && A->Indices[i][j]==i && A->Values[i][j]!=0){
				S->Inv_diag[i] = (float)1./A->Values[i][j];			
				S->Val[offset+j] = 0.;												// subtract diagonal 
			}
		}
	}
}

void sparse_to_csr_device(csr_matrix* _S, sparse_matrix* A, int store_inv_diag){
	cudaError_t status1,status2,status3;
	cudaError_t status4 = cudaSuccess;
	csr_matrix a;
	convert_sparse_to_csr(&a,A,store_inv_diag);
	_S->total_size = a.total_size;
	_S->row_num = a.row_num;
	status1 = cudaMalloc((void**)&(_S->Start_index),(_S->row_num+1)*sizeof(int));
	status2 = cudaMalloc((void**)&(_S->Ind),(_S->total_size)*sizeof(int));
	status3 = cudaMalloc((void**)&(_S->Val),(_S->total_size)*sizeof(float));
	if (store_inv_diag) status4 = cudaMalloc((void**)&(_S->Inv_diag),(_S->row_num)*sizeof(float));
	if (status1!=cudaSuccess || status2!=cudaSuccess || status3!=cudaSuccess || status4!=cudaSuccess){
		printf("error allocating sparse matrix on device -> abort\n");
		exit(0);
	}
	status4 = cudaSuccess;
	status1 = cudaMemcpy(_S->Start_index,a.Start_index,(_S->row_num+1)*sizeof(int),cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(_S->Ind,a.Ind,(_S->total_size)*sizeof(int),cudaMemcpyHostToDevice);
	status3 = cudaMemcpy(_S->Val,a.Val,(_S->total_size)*sizeof(float),cudaMemcpyHostToDevice);
	if (store_inv_diag) status4 = cudaMemcpy(_S->Inv_diag,a.Inv_diag,(_S->row_num)*sizeof(float),cudaMemcpyHostToDevice);
	if (status1!=cudaSuccess || status2!=cudaSuccess || status3!=cudaSuccess || status4!=cudaSuccess){
		printf("error cpoying sparse matrix to device -> abort\n");
		exit(0);
	}
	free(a.Start_index);
	free(a.Ind);
	free(a.Val);
}

void free_csr_device(csr_matrix* _S){
	cudaFree(_S->Start_index);
	cudaFree(_S->Ind);
	cudaFree(_S->Val);
}

void print_dev_prop(){
	int i;
	cudaDeviceProp prop;
	int num = 0;
	cudaGetDeviceCount(&num);
	printf("number of devices: %d\n",num);
	for (i=0;i<num;i++){
		cudaGetDeviceProperties(&prop,i);
		printf("name of device %d: %s\n",i+1,prop.name);
		printf("revision: %d.%d\n",prop.major,prop.minor);
		printf("global memory: %d MB\n",prop.totalGlobalMem/(1024*1024));
		printf("block grid sizes: %dx%dx%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
		printf("max threads per block: %d\n",prop.maxThreadsPerBlock);
		printf("number of processors: %d\n",prop.multiProcessorCount);
		printf("shared memory per block: %d doubles\n",prop.sharedMemPerBlock/sizeof(double));
		printf("warp size: %d\n",prop.warpSize);
		printf("\n");
	}
}

cudaError_t cpy_df_to_device(float* _X,double* X,int n){
	cudaError_t status;
	int i;
	float* Y = (float*)malloc(n*sizeof(float));
	for (i=0;i<n;i++) Y[i] = (float)X[i];
	status = cudaMemcpy(_X,Y,n*sizeof(float),cudaMemcpyHostToDevice);
	free(Y);
	return status;
}

cudaError_t cpy_fd_to_host(double* X,float* _X,int n){
	cudaError_t status;
	int i;
	float* Y = (float*)malloc(n*sizeof(float));
	status = cudaMemcpy(Y,_X,n*sizeof(float),cudaMemcpyDeviceToHost);
	for (i=0;i<n;i++) X[i] = (double)Y[i];
	free(Y);
	return status;
}

__global__ void csv(float* _X,int n){
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	while (thr_index<n){
		_X[thr_index] = 0;
		thr_index += blockDim.x*gridDim.x;
	}
}

float* cuda_zero_vector(int n){
	float* _Z = NULL;
	cudaMalloc((void**)&_Z,n*sizeof(float));
	csv<<<blocks,threads>>>(_Z,n);
	return _Z;
}

float* cuda_get_vector_on_device(double* X,int n){
	cudaError_t status1,status2;
	float* _X = NULL;
	status1 = cudaMalloc((void**)&_X,n*sizeof(float));
	status2 = cpy_df_to_device(_X,X,n);
	if (status1!=cudaSuccess || status2!=cudaSuccess){
		printf("cuda_get_vector_on_device: failed -> abort\n");
		exit(0);
	}
	return _X;
}

__global__ void cuda_pseudo_mult(float* _X,float* _Y,int n){
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	while (thr_index<n){
		_X[thr_index] *= _Y[thr_index];
		thr_index += blockDim.x*gridDim.x;
	}
}

__global__ void cuda_vector_add(float* X,float* Y,double factor,int n){			// x+factor*y->x
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	while (thr_index<n){
		X[thr_index] += factor*Y[thr_index];
		thr_index += blockDim.x*gridDim.x;
	}
}

__global__ void cuda_vector_mult_add(float* X,float* Y,float factor,int n){			// factor*x+y->x
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	while (thr_index<n){
		X[thr_index] = factor*X[thr_index]+Y[thr_index];
		thr_index += blockDim.x*gridDim.x;
	}
}

__global__ void csc(float* _X, float* _Y,int n,float* Partial_sums){
	extern __shared__ float Block_sums[];
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	float sum = 0;
	while (thr_index<n){
		sum += _X[thr_index]*_Y[thr_index];
		thr_index += blockDim.x*gridDim.x;
	}
	int m = threadIdx.x;
	Block_sums[m] = sum;
	
	__syncthreads();
	
	int i = blockDim.x/2;
	int odd = blockDim.x % 2;
	while (i!=0){
		if (m<i) Block_sums[m] += Block_sums[m+i];
		if (m==0 && odd) Block_sums[0] += Block_sums[2*i];
		__syncthreads();
		odd = i % 2;
		i /= 2;
	}
	
	if (m==0) Partial_sums[blockIdx.x] = Block_sums[0];
}

float cuda_scalar(float* _X, float* _Y,int n){			// Hier Fehler bei malloc blocks -> n ? 
	int i;
	double sum = 0;
	double* Z = zero_vector(blocks);
	float* _Z = NULL;
	cudaMalloc((void**)&_Z,blocks*sizeof(float));
	cpy_df_to_device(_Z,Z,blocks);
	//cudaMemcpy(_Z,Z,blocks*sizeof(float),cudaMemcpyHostToDevice);
	csc<<<blocks,threads,threads*sizeof(float)>>>(_X,_Y,n,_Z);
	//cudaMemcpy(Z,_Z,blocks*sizeof(float),cudaMemcpyDeviceToHost);
	cpy_fd_to_host(Z,_Z,blocks);
	for (i=0;i<blocks;i++) sum += Z[i]; 
	cudaFree(_Z);
	free(Z);
	return sum;
}

float cuda_euklid_norm(float* _X,int n){
	float res = cuda_scalar(_X,_X,n);
	return sqrt(res);
}

void cuda_sparse_on_device(sparse_matrix* A,int max_bandwidth,int** _Indices,float** _Values){
	int i,j,k;
	cudaError_t error_code;
	int n = A->size;
	int N = n*max_bandwidth;
	int* Indices = (int*)malloc(N*sizeof(int));
	double* Values = (double*)malloc(N*sizeof(double));
	for (i=0;i<n;i++){		
		for(j=0;j<max_bandwidth;j++){
			k = i*max_bandwidth+j;
			if (j<A->Len[i]){				
				Indices[k] = A->Indices[i][j]; 
				Values[k] = A->Values[i][j];
			}
			else{
				Indices[k] = -1;
				Values[k] = 0;
			}
		}
	}
	printf("maximum bandwidth: %d\n",max_bandwidth);
	error_code = cudaMalloc((void**)_Indices,N*sizeof(int));
	if (error_code!=cudaSuccess) printf("error in cuda_sparse_on_device:cudaMalloc: %s\n",cudaGetErrorString(error_code));
	error_code = cudaMemcpy(*_Indices,Indices,N*sizeof(int),cudaMemcpyHostToDevice);
	if (error_code!=cudaSuccess) printf("error in cuda_sparse_on_device:cudaMemcpy %s\n",cudaGetErrorString(error_code));
	error_code = cudaMalloc((void**)_Values,N*sizeof(float));
	if (error_code!=cudaSuccess) printf("error in cuda_sparse_on_device:cudaMalloc: %s\n",cudaGetErrorString(error_code));
	error_code = cpy_df_to_device(*_Values,Values,N);
	if (error_code!=cudaSuccess) printf("error in cuda_sparse_on_device:cudaMemcpy %s\n",cudaGetErrorString(error_code));
	
	free(Indices);
	free(Values);
}

__global__ void cuda_jacobi_iterate(float* _X,float* _Xprev, float* _b,int* _Indices,float* _Values,int n,int max_bandwidth){
	int thr_index = threadIdx.x+blockIdx.x*blockDim.x;
	int i,ind;
	float sum,aii;
	while (thr_index<n){
		sum = _b[thr_index];
		aii = 0;
		i = thr_index*max_bandwidth;
		while(i<(thr_index+1)*max_bandwidth){
			ind = _Indices[i];
			if (ind<0) break;
			else{
				if (ind==thr_index) aii = _Values[i];
				else sum -= _Values[i]*_Xprev[ind];
			}
			i++;
		}
		if (aii!=0) _X[thr_index] = sum/aii; else _X[thr_index] = NAN;
		thr_index += blockDim.x*gridDim.x;
	}
}

/*void cuda_jacobi(double* X,double* b,int* _Indices,double* _Values,int n,int max_bandwidth,int iter){			// jocobi method
	int i;
	double* _X = NULL;
	double* _Xprev = NULL;
	double* _b = NULL;
	cudaMalloc((void**)&_X,n*sizeof(double));
	cudaMalloc((void**)&_Xprev,n*sizeof(double));
	cudaMalloc((void**)&_b,n*sizeof(double));
	cudaMemcpy(_X,X,n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(_b,b,n*sizeof(double),cudaMemcpyHostToDevice);
	
	for (i=0;i<iter;i++){
		
		cudaMemcpy(_Xprev,_X,n*sizeof(double),cudaMemcpyDeviceToDevice);
		cuda_jacobi_iterate<<<blocks,threads>>>(_X,_Xprev,_b,_Indices,_Values,n,max_bandwidth);		
		
	}
	
	cudaMemcpy(X,_X,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(_b);
	cudaFree(_X);
	cudaFree(_Xprev);
}*/

void cuda_csr_jacobi(csr_matrix* _S,float* _X,float* _b,int iter){
	int i;
	cudaError_t cstatus;
	cusparseStatus_t status;
	float* _Y = NULL;
	
	float alpha = -1.;
	float beta = 1.;
	int n = _S->row_num;
	
	cstatus = cudaMalloc((void**)&_Y,n*sizeof(float));
	if (cstatus!=cudaSuccess){
		printf("error allocating sparse matrix on device -> abort\n");
		exit(0);
	}
	
	device_print_vector(_S->Val,_S->total_size);
	
	for (i=0;i<iter;i++){
		
		//status = cusparseDcsrmv(cL_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,n,-1.0,cL_descra,_S->Val,_S->Ind,_S->Start_index,_X,1.0,_Y);		
		status = cusparseScsrmv(cL_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,n,_S->total_size,&alpha,cL_descra,_S->Val,_S->Ind,_S->Start_index,_X,&beta,_Y);
		cuda_pseudo_mult<<<blocks,threads>>>(_Y,_S->Inv_diag,n);
		
		cudaMemcpy(_X,_Y,n*sizeof(float),cudaMemcpyDeviceToDevice);
		
	}
	
	cudaFree(_Y);
}
	
void cuda_gpu_settings(int deviceID){
	cudaError_t status;
	cusparseStatus_t status1,status2;
	
	// general cuda sttings
	print_dev_prop();
	blocks = 8;
	threads = 128;
	cudaSetDevice(deviceID);
	status = cudaDeviceReset();
	if (status!=cudaSuccess) printf("failed to reset device %d\n",deviceID);
	
	// cuSPARSE settings
	status1 = cusparseCreate(&cL_handle);
	status2 = cusparseCreateMatDescr(&cL_descra);
	if (status1!=CUSPARSE_STATUS_SUCCESS || status2!=CUSPARSE_STATUS_SUCCESS){
		printf("failed to initialize cuSPARSE -> abort\n");
		exit(0);
	}
	
	status1 = cusparseSetMatType(cL_descra,CUSPARSE_MATRIX_TYPE_GENERAL);
	status2 = cusparseSetMatIndexBase(cL_descra,CUSPARSE_INDEX_BASE_ZERO);
	if (status1!=CUSPARSE_STATUS_SUCCESS || status2!=CUSPARSE_STATUS_SUCCESS){
		printf("failed to initialize cuSPARSE matrix -> abort\n");
		exit(0);
	}
	
	
}


/*void cuda_PCG_solve(sparse_matrix* A,double* b,double* X){
	double scl,alpha,beta;
	int n = A->size;
	double* _X = cuda_get_vector_on_device(X,n);
	double* _B = cuda_get_vector_on_device(b,n);
	sparse_matrix* _A = cuda_get_sparse_on_device(A);
	
	double* _R = cuda_zero_vector(n);
	double* _Z = cuda_zero_vector(n);
	
	cuda_linear_map<<<blocks,threads>>>(*_A,_B,_X,-1.0,_R);
	(*cuda_CG_Precon)(_Z,_R);
	double* _P = cuda_get_vector_on_device(_Z,n);
	double* _AP = cuda_zero_vector(n);
	int iter = 0;
	double res = cuda_euklid_norm(_Z,n);
	while (res>CG_eps && iter<CG_max_iter){
		cuda_sparse_mult<<<blocks,threads>>>(*_A,_P,_AP);
		scl = cuda_scalar(_R,_Z,n);
		alpha = scl/cuda_scalar(_P,_AP,n);
		cuda_vector_add<<<blocks,threads>>>(_X,_P,alpha,n);
		cuda_vector_add<<<blocks,threads>>>(_R,_AP,-alpha,n);
		(*cuda_CG_Precon)(_Z,_R);
		beta = 1./scl;
		scl = cuda_scalar(_R,_Z,n);
		beta *= scl;
		cuda_vector_mult_add<<<blocks,threads>>>(_P,_Z,beta,n);
		res = cuda_euklid_norm(_Z,n);
		iter++;
	}
	if (iter==CG_max_iter) printf("Warning: PCG did not converge within %d iterations\n",iter); else printf("PCG converged in %d iterations\n",iter);
	
	cuda_sparse_free(_A);
	cudaFree(_A);
	cudaFree(_X);
	cudaFree(_B);
}*/
