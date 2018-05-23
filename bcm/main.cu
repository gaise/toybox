#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

inline void gassert(cudaError_t err_code, const char *file, int line)
{
	if (err_code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val) gassert(val, __FILE__, __LINE__)

__global__ void buildClusterMatrixNeo(float *x, float *y, float *z, int *cluster_indices, int *cluster_matrix, int *cluster_offset, int size, int cluster_num, float threshold)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockIdx.x * gridDim.x;

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

__global__ void buildClusterMatrix(float *x, float *y, float *z, int *cluster_indices, int *cluster_matrix, int *cluster_offset, int size, int cluster_num, float threshold)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ float local_x[1024];
	__shared__ float local_y[1024];
	__shared__ float local_z[1024];

	for (int column = index; column < size; column += stride) {
		local_x[threadIdx.x] = x[column];
		local_y[threadIdx.x] = y[column];
		local_z[threadIdx.x] = z[column];
		int column_cluster = cluster_indices[column];
		int cc_offset = cluster_offset[column_cluster];

		__syncthreads();

		for (int row = 0; row < column; row++) {
			float tmp_x = x[row] - local_x[threadIdx.x];
			float tmp_y = y[row] - local_y[threadIdx.x];
			float tmp_z = z[row] - local_z[threadIdx.x];
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
	int i, mode;
	char filename[128];
	
    std::cout << "input filename..." << std::endl;
	std::cin >> filename;

    std::cout << "select function: 1. original  2. improved..." << std::endl;
    std::cin >> mode;
    
	FILE *fp = fopen(filename, "r");

	char readline[128];

	int size, cluster_num;
	float threshold;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    double time;

	std::cout << "start." << std::endl;

	fgets(readline, sizeof(char)*128, fp);
	size = atoi(readline);

	std::cout << "read size: " << size << std::endl;
	
	fgets(readline, sizeof(char)*128, fp);
	cluster_num = atoi(readline);

	std::cout << "read cluster_num: " << cluster_num << std::endl;

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
		h_x[i] = (float)atoi(readline);
		fgets(readline, sizeof(char)*128, fp);
		h_y[i] = (float)atoi(readline);
		fgets(readline, sizeof(char)*128, fp);
		h_z[i] = (float)atoi(readline);
	}

	std::cout << "read x, y, z." << std::endl;

	int *h_indices, *h_offset;
	h_indices = (int*)malloc(size * sizeof(int));
	h_offset = (int*)malloc(size * sizeof(int));

	for (i = 0; i < size; i++) {
		fgets(readline, sizeof(char)*128, fp);
		h_indices[i] = atoi(readline);
		fgets(readline, sizeof(char)*128, fp);
		h_offset[i] = atoi(readline);
	}

	std::cout << "read indices, offset." << std::endl;

	float *dev_x, *dev_y, *dev_z;
	checkCudaErrors(cudaMalloc((void**)&dev_x, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_y, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_z, size * sizeof(float)));
	
	checkCudaErrors(cudaMemcpy(dev_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z, h_z, size * sizeof(float), cudaMemcpyHostToDevice));

	int *dev_indices, *dev_offset, *dev_matrix, *dev_matrix2;
	checkCudaErrors(cudaMalloc((void**)&dev_indices, size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_offset, size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_matrix, cluster_num * cluster_num * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_matrix2, cluster_num * cluster_num * sizeof(int)));

	checkCudaErrors(cudaMemcpy(dev_indices, h_indices, size * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_offset, h_offset, size * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(dev_matrix, 0, cluster_num * cluster_num * sizeof(int)));
    checkCudaErrors(cudaMemset(dev_matrix2, 0, cluster_num * cluster_num * sizeof(int)));

	int block_x, grid_x;
	block_x = (size > 1024) ? 1024 : size;
	grid_x = (size - 1) / block_x + 1;

    if (mode == 1) {
        start = std::chrono::system_clock::now();
 
	    buildClusterMatrix<<<grid_x, block_x>>>(dev_x, dev_y, dev_z, dev_indices, dev_matrix, dev_offset, size, cluster_num, threshold);

	    checkCudaErrors(cudaDeviceSynchronize());

        end = std::chrono::system_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "original elapsed time: " << time << "ms." << std::endl;
    }
    else {
        start = std::chrono::system_clock::now();

        buildClusterMatrixP<<<grid_x, block_x>>>(dev_x, dev_y, dev_z, dev_indices, dev_matrix2, dev_offset, size, cluster_num, threshold);
        checkCudaErrors(cudaDeviceSynchronize());

        end = std::chrono::system_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "improved elapsed time: " << time << "ms." << std::endl;
    }
    int *h_matrix, *h_matrix2;
    h_matrix = (int*)malloc(cluster_num*cluster_num*sizeof(int));
    h_matrix2 = (int*)malloc(cluster_num*cluster_num*sizeof(int));

    checkCudaErrors(cudaMemcpy(h_matrix, dev_matrix, cluster_num*cluster_num*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_matrix2, dev_matrix2, cluster_num*cluster_num*sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < cluster_num*cluster_num; i++) {
    //     if (h_matrix[i] != h_matrix2[i]) std::cout << "wrong: " << i << std::endl;
    // }

	checkCudaErrors(cudaFree(dev_x));
	checkCudaErrors(cudaFree(dev_y));
	checkCudaErrors(cudaFree(dev_z));
	checkCudaErrors(cudaFree(dev_offset));
	checkCudaErrors(cudaFree(dev_indices));
	checkCudaErrors(cudaFree(dev_matrix));

	free(h_x);
	free(h_y);
	free(h_z);
	free(h_indices);
	free(h_offset);
	free(h_matrix);

	return 0;
}
