#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

#define BLOCK_SIZE_X 1024

inline void gassert(cudaError_t err_code, const char *file, int line)
{
	if (err_code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val) gassert(val, __FILE__, __LINE__)

__global__ void pclEuclideanInitialize(int *cluster_indices, int size)
{
    for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x)
        cluster_indices[index] = index;
}

__global__ void blockLabelling(float *x, float *y, float *z, int *cluster_indices, int size, float threshold)
{
	int block_start = blockIdx.x * blockDim.x;
	int block_end = (block_start + blockDim.x <= size) ? (block_start + blockDim.x) : size;
	int row = threadIdx.x + block_start;
	__shared__ int local_offset[BLOCK_SIZE_X];
	__shared__ float local_x[BLOCK_SIZE_X];
	__shared__ float local_y[BLOCK_SIZE_X];
	__shared__ float local_z[BLOCK_SIZE_X];
	__shared__ int local_cluster_changed[BLOCK_SIZE_X];

	if (row < block_end) {
		local_offset[threadIdx.x] = threadIdx.x;
		local_x[threadIdx.x] = x[row];
		local_y[threadIdx.x] = y[row];
		local_z[threadIdx.x] = z[row];
		__syncthreads();

		for (int column = block_start; column < block_end; column++) {
			float tmp_x = local_x[threadIdx.x] - local_x[column - block_start];
			float tmp_y = local_y[threadIdx.x] - local_y[column - block_start];
			float tmp_z = local_z[threadIdx.x] - local_z[column - block_start];
			int column_offset = local_offset[column - block_start];
			int row_offset = local_offset[threadIdx.x];

			local_cluster_changed[threadIdx.x] = 0;
			__syncthreads();

			if (row > column && column_offset != row_offset && norm3df(tmp_x, tmp_y, tmp_z) < threshold)
				local_cluster_changed[row_offset] = 1;
			__syncthreads();

			local_offset[threadIdx.x] = (local_cluster_changed[row_offset] == 1) ? column_offset : row_offset;
			__syncthreads();
		}

		__syncthreads();

		int new_cluster = cluster_indices[block_start + local_offset[threadIdx.x]];

		__syncthreads();

		cluster_indices[row] = new_cluster;
	}
}

__global__ void clusterMark(int *cluster_list, int *cluster_mark, int size)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x)
		cluster_mark[cluster_list[i]] = 1;
}

void exclusiveScan(int *input, int ele_num, int *sum)
{
	thrust::device_ptr<int> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

__global__ void clusterCollector(int *old_cluster_list, int *new_cluster_list, int *cluster_location, int size)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x)
		new_cluster_list[cluster_location[old_cluster_list[i]]] = old_cluster_list[i];
}

__global__ void buildClusterMatrix(float *x, float *y, float *z, int *cluster_indices, int *cluster_matrix, int *cluster_offset, int size, int cluster_num, float threshold)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float local_x;
	float local_y;
	float local_z;

	for (int column = index; column < size; column += stride) {
		local_x = x[column];
		local_y = y[column];
		local_z = z[column];
		int column_cluster = cluster_indices[column];
		int cc_offset = cluster_offset[column_cluster];

		__syncthreads();

		for (int row = 0; row < column; row++) {
			float tmp_x = x[row] - local_x;
			float tmp_y = y[row] - local_y;
			float tmp_z = z[row] - local_z;
			int row_cluster = cluster_indices[row];
			int rc_offset = cluster_offset[row_cluster];

			__syncthreads();

			if (row_cluster != column_cluster && norm3df(tmp_x, tmp_y, tmp_z) < threshold)
				cluster_matrix[rc_offset * cluster_num + cc_offset] = 1;
		}
	}
}

__global__ void buildClusterMatrixP(float *x, float *y, float *z, int *cluster_indices, int *cluster_matrix, int *cluster_offset, int size, int cluster_num, float threshold)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    register float local_x, local_y, local_z;

    int iter_num = size / 2;

    for (int column = index; column < size; column += stride) {
        local_x = x[column];
        local_y = y[column];
        local_z = z[column];
        int column_cluster = cluster_indices[column];
        int cc_offset = cluster_offset[column_cluster];

        int thindex = column - 1;

        __syncthreads();

        for (int row = 0; row < iter_num; row++) {
            if (thindex < 0) thindex = size - 1;

            float tmp_x = x[thindex] - local_x;
            float tmp_y = y[thindex] - local_y;
            float tmp_z = z[thindex] - local_z;
            int row_cluster = cluster_indices[thindex];
            int rc_offset = cluster_offset[row_cluster];

            __syncthreads();

            if (row_cluster != column_cluster && norm3df(tmp_x, tmp_y, tmp_z) < threshold)
                cluster_matrix[rc_offset * cluster_num + cc_offset] = 1;

            thindex--;
        }
    }
}

int main()
{
	int i;
	
	FILE *fp = fopen("/home/autoware/bcm_param/bcm_param_0.txt", "r");

	char readline[128];

	int size;
	float threshold;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    double time;

	fgets(readline, sizeof(char)*128, fp);
	size = atoi(readline);

	std::cout << "read size: " << size << std::endl;
	
	fgets(readline, sizeof(char)*128, fp);
	threshold = atof(readline);

	std::cout << "read threshold: " << threshold << std::endl;

	float *h_x, *h_y, *h_z;
	h_x = (float*)malloc(size * sizeof(float)); 
	h_y = (float*)malloc(size * sizeof(float)); 
	h_z = (float*)malloc(size * sizeof(float)); 

	for (i = 0; i < size; i++) {
		std::cout << "%d...\r";
		fgets(readline, sizeof(char)*128, fp);
		h_x[i] = atof(readline);
		fgets(readline, sizeof(char)*128, fp);
		h_y[i] = atof(readline);
		fgets(readline, sizeof(char)*128, fp);
		h_z[i] = atof(readline);
	}

	std::cout << "read x, y, z." << std::endl;
	
    float *dev_x, *dev_y, *dev_z;
	checkCudaErrors(cudaMalloc((void**)&dev_x, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_y, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_z, size * sizeof(float)));
	
	checkCudaErrors(cudaMemcpy(dev_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z, h_z, size * sizeof(float), cudaMemcpyHostToDevice));

    int block_x, grid_x;
    
    block_x = (size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : size;
    grid_x = (size - 1) / block_x + 1;

    start = std::chrono::system_clock::now();

    int *cluster_offset;
    int cluster_num, old_cluster_num;

    int *cluster_indices;
    checkCudaErrors(cudaMalloc((void**)&cluster_indices, size * sizeof(int)));

    pclEuclideanInitialize<<<grid_x, block_x>>>(cluster_indices, size);
    checkCudaErrors(cudaDeviceSynchronize());

    old_cluster_num = cluster_num = size;

    checkCudaErrors(cudaMalloc(&cluster_offset, (size + 1) * sizeof(int)));
    checkCudaErrors(cudaMemset(cluster_offset, 0, (size + 1) * sizeof(int)));
    blockLabelling<<<grid_x, block_x>>>(dev_x, dev_y, dev_z, cluster_indices, size, threshold);
    clusterMark<<<grid_x, block_x>>>(cluster_indices, cluster_offset, size);
    exclusiveScan(cluster_offset, size + 1, &cluster_num);

    int *cluster_list, *new_cluster_list, *tmp;

    checkCudaErrors(cudaMalloc(&cluster_list, cluster_num * sizeof(int)));
    clusterCollector<<<grid_x, block_x>>>(cluster_indices, cluster_list, cluster_offset, size);
    checkCudaErrors(cudaDeviceSynchronize());

    int *cluster_matrix;
    int *new_cluster_matrix;

    checkCudaErrors(cudaMalloc((void**)&cluster_matrix, cluster_num * cluster_num * sizeof(int)));
    checkCudaErrors(cudaMemset(cluster_matrix, 0, cluster_num * cluster_num * sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc(&new_cluster_list, cluster_num * sizeof(int)));

    start = std::chrono::system_clock::now();
 
	buildClusterMatrixP<<<grid_x, block_x>>>(dev_x, dev_y, dev_z, cluster_indices, cluster_matrix, cluster_offset, size, cluster_num, threshold);

	checkCudaErrors(cudaDeviceSynchronize());

    end = std::chrono::system_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "original elapsed time: " << time << "ms." << std::endl;
    
    int *h_matrix;
    h_matrix = (int*)malloc(cluster_num*cluster_num*sizeof(int));

    checkCudaErrors(cudaMemcpy(h_matrix, cluster_matrix, cluster_num*cluster_num*sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(dev_x));
	checkCudaErrors(cudaFree(dev_y));
	checkCudaErrors(cudaFree(dev_z));
	checkCudaErrors(cudaFree(cluster_offset));
	checkCudaErrors(cudaFree(cluster_indices));
	checkCudaErrors(cudaFree(cluster_matrix));
    checkCudaErrors(cudaFree(cluster_list));
    checkCudaErrors(cudaFree(new_cluster_list));

	free(h_x);
	free(h_y);
	free(h_z);
	free(h_matrix);

	return 0;
}
