#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 300000

#define checkCudaErrors(msg) err_msg(msg, __LINE__)

void err_msg(cudaError_t msg, int x)
{
 if (msg != cudaSuccess) {
   std::cerr << "In line: " << x << ". error: " << cudaGetErrorString(msg) << std::endl;
   exit(1);
 }

 return;
}

// void debug(int *x) {
// 	int *h_x = new float[N];
// 	checkCudaErrors(cudaMemcpy(h_x, x, sizeof(float)*N, cudaMemcpyDeviceToHost));
	
// 	for (float i = 0; i < N; i++) {
// 		std::cout << h_x[i] << " ";
// 	}
// 	std::cout << std::endl;

// 	delete[] h_x;
// }

__global__ void seperateMaxMin(float *x, float *y, float *z, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float tmp;

	for (int i = index; i < half_size; i += stride) {
		if (i + half_size >= full_size) break;
		
		if (x[i] < x[i+half_size])
			{
				tmp = x[i];
				x[i] = x[i+half_size];
				x[i+half_size] = tmp;
			}

		if (y[i] < y[i+half_size])
			{
				tmp = y[i];
				y[i] = y[i+half_size];
				y[i+half_size] = tmp;
			}

		if (z[i] < z[i+half_size])
			{
				tmp = z[i];
				z[i] = z[i+half_size];
				z[i+half_size] = tmp;
			}
	}
}

__global__ void findMax(float *x, float *y, float *z, int full_size, int half_size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < half_size; i += stride) {
    x[i] = (i + half_size < full_size) ? ((x[i] >= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
    y[i] = (i + half_size < full_size) ? ((y[i] >= y[i + half_size]) ? y[i] : y[i + half_size]) : y[i];
    z[i] = (i + half_size < full_size) ? ((z[i] >= z[i + half_size]) ? z[i] : z[i + half_size]) : z[i];
  }
}

__global__ void findMin(float *x, float *y, float *z, int full_size, int half_size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < half_size; i += stride) {
    x[i] = (i + half_size < full_size) ? ((x[i] <= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
    y[i] = (i + half_size < full_size) ? ((y[i] <= y[i + half_size]) ? y[i] : y[i + half_size]) : y[i];
    z[i] = (i + half_size < full_size) ? ((z[i] <= z[i + half_size]) ? z[i] : z[i + half_size]) : z[i];
  }
}

int main() 
{
	float *max_x, *max_y, *max_z, *x, *y, *z, *d_x, *d_y, *d_z;
	float x_max, x_min, y_max, y_min, z_max, z_min;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	double time;

	x = new float[N];
	y = new float[N];
	z = new float[N];

	std::mt19937 mt(10);

	for (int i = 0; i < N; i++) {
		x[i] = mt() / 100000.0;
		y[i] = mt() / 100000.0;
		z[i] = mt() / 100000.0;

		if (i == 0) 
			{
				x_max = x_min = x[i];
				y_max = y_min = y[i];
				z_max = z_min = z[i];
			}
		else
			{
				x_max = (x_max >= x[i]) ? x_max : x[i];
				x_min = (x_min <= x[i]) ? x_min : x[i];
				y_max = (y_max >= y[i]) ? y_max : y[i];
				y_min = (y_min <= y[i]) ? y_min : y[i];
				z_max = (z_max >= z[i]) ? z_max : z[i];
				z_min = (z_min <= z[i]) ? z_min : z[i];
			}
	}

	int points_num = N;

	std::cout << "correct x max: " << x_max << std::endl;
	std::cout << "correct x min: " << x_min << std::endl;
	std::cout << "correct y max: " << y_max << std::endl;
	std::cout << "correct y min: " << y_min << std::endl;
	std::cout << "correct z max: " << z_max << std::endl;
	std::cout << "correct z min: " << z_min << std::endl;

	checkCudaErrors(cudaMalloc(&d_x, sizeof(float) * points_num));
	checkCudaErrors(cudaMalloc(&d_y, sizeof(float) * points_num));
	checkCudaErrors(cudaMalloc(&d_z, sizeof(float) * points_num));

	checkCudaErrors(cudaMemcpy(d_x, x, sizeof(float) * points_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y, y, sizeof(float) * points_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_z, z, sizeof(float) * points_num, cudaMemcpyHostToDevice));

	start = std::chrono::system_clock::now();

	checkCudaErrors(cudaMalloc((void**)&max_x, sizeof(float)*points_num));
	checkCudaErrors(cudaMalloc((void**)&max_y, sizeof(float)*points_num));
	checkCudaErrors(cudaMalloc((void**)&max_z, sizeof(float)*points_num));
	checkCudaErrors(cudaMemcpy(max_x, d_x, sizeof(float)*points_num, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_y, d_y, sizeof(float)*points_num, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_z, d_z, sizeof(float)*points_num, cudaMemcpyDeviceToDevice));

	// debug(max_x);

	int half_points_num = (points_num - 1) / 2 + 1;
	int block_x = (half_points_num > 1024) ? 1024 : half_points_num;
	int grid_x = (half_points_num - 1) / block_x + 1;

	seperateMaxMin<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num, half_points_num);
	checkCudaErrors(cudaGetLastError());
	
	// debug(max_x);

	int min_points_num = half_points_num;
	int min_half_points_num;
	float *min_x = max_x + (points_num / 2);
	float *min_y = max_y + (points_num / 2);
	float *min_z = max_z + (points_num / 2);
	points_num = half_points_num;

	while (points_num > 1) {
		half_points_num = (points_num - 1) / 2 + 1;
		block_x = (half_points_num > 1024) ? 1024 : half_points_num;
		grid_x = (half_points_num - 1) / block_x + 1;
		
		findMax<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num, half_points_num);
		checkCudaErrors(cudaGetLastError());

		points_num = half_points_num;
	}
	
	while (min_points_num > 1) {
		min_half_points_num = (min_points_num - 1) / 2 + 1;
		block_x = (min_half_points_num > 1024) ? 1024 : min_half_points_num;
		grid_x = (min_half_points_num - 1) / block_x + 1;

		findMin<<<grid_x, block_x>>>(min_x, min_y, min_z, min_points_num, min_half_points_num);
		checkCudaErrors(cudaGetLastError());

		min_points_num = min_half_points_num;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&x_max, max_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&x_min, min_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&y_max, max_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&y_min, min_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&z_max, max_z, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&z_min, min_z, sizeof(float), cudaMemcpyDeviceToHost));

	end = std::chrono::system_clock::now();
	time =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

	std::cout << "GPU x max: " << x_max << std::endl;
	std::cout << "GPU x min: " << x_min << std::endl;
	std::cout << "GPU y max: " << y_max << std::endl;
	std::cout << "GPU y min: " << y_min << std::endl;
	std::cout << "GPU z max: " << z_max << std::endl;
	std::cout << "GPU z min: " << z_min << std::endl;

	std::cout << "time: " << time << "ms." << std::endl;

	checkCudaErrors(cudaFree(max_x));
	checkCudaErrors(cudaFree(max_y));
	checkCudaErrors(cudaFree(max_z));
	
	delete[] x;
	delete[] y;
	delete[] z;

	return 0;
}