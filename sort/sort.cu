#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>

#define N 300000
#define BLOCK_SIZE_X 1024

#define checkCudaErrors(msg) err_msg(msg, __LINE__)

void err_msg(cudaError_t msg, int x)
{
 if (msg != cudaSuccess) {
   std::cerr << "In line: " << x << ". error: " << cudaGetErrorString(msg) << std::endl;
   exit(1);
 }

 return;
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
  int points_num = N;
  float *x, *y, *z;
  float *max_x, *max_y, *max_z, *min_x, *min_y, *min_z;
  float a_max_x, a_min_x, a_max_y, a_min_y, a_max_z, a_min_z;
  float *d_x, *d_y, *d_z;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  double time;

  x = new float[N];
  y = new float[N];
  z = new float[N];

  std::mt19937 mt(10);

  for (int i = 0; i < N; i++) {
    x[i] = mt() / 10000.0;
    y[i] = mt() / 10000.0;
    z[i] = mt() / 10000.0;
  }

  
  start = std::chrono::system_clock::now();

  a_max_x = a_min_x = x[0];
  a_max_y = a_min_y = y[0];
  a_max_z = a_min_z = z[0];

  for (int i = 1; i < N; i++) {
    a_max_x = (a_max_x > x[i]) ? a_max_x : x[i];
    a_min_x = (a_min_x < x[i]) ? a_min_x : x[i];

    a_max_y = (a_max_y > y[i]) ? a_max_y : y[i];
    a_min_y = (a_min_y < y[i]) ? a_min_y : y[i];

    a_max_z = (a_max_z > z[i]) ? a_max_z : z[i];
    a_min_z = (a_min_z < z[i]) ? a_min_z : z[i];
  }

  end = std::chrono::system_clock::now();

  time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  std::cout << "CPU sort: " << time << "ms." << std::endl;

  checkCudaErrors(cudaMalloc(&d_x, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&d_y, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&d_z, sizeof(float) * points_num));

  checkCudaErrors(cudaMemcpy(d_x, x, sizeof(float) * points_num, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y, sizeof(float) * points_num, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_z, z, sizeof(float) * points_num, cudaMemcpyHostToDevice));

  start = std::chrono::system_clock::now();

  checkCudaErrors(cudaMalloc(&max_x, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&max_y, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&max_z, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&min_x, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&min_y, sizeof(float) * points_num));
  checkCudaErrors(cudaMalloc(&min_z, sizeof(float) * points_num));

  checkCudaErrors(cudaMemcpy(max_x, d_x, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(max_y, d_y, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(max_z, d_z, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemcpy(min_x, d_x, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(min_y, d_y, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(min_z, d_z, sizeof(float) * points_num, cudaMemcpyDeviceToDevice));

  while (points_num > 1) {
    int half_points_num = (points_num - 1) / 2 + 1;
    int block_x = (half_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_points_num;
    int grid_x = (half_points_num - 1) / block_x + 1;

    findMax<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num, half_points_num);
    checkCudaErrors(cudaGetLastError());

    findMin<<<grid_x, block_x>>>(min_x, min_y, min_z, points_num, half_points_num);
    checkCudaErrors(cudaGetLastError());

    points_num = half_points_num;
  }

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(&a_max_x, max_x, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&a_max_y, max_y, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&a_max_z, max_z, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(&a_min_x, min_x, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&a_min_y, min_y, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&a_min_z, min_z, sizeof(float), cudaMemcpyDeviceToHost));

  end = std::chrono::system_clock::now();

  time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  std::cout << "GPU sort: " << time << "ms." << std::endl;

  checkCudaErrors(cudaFree(max_x));
  checkCudaErrors(cudaFree(max_y));
  checkCudaErrors(cudaFree(max_z));
  checkCudaErrors(cudaFree(min_x));
  checkCudaErrors(cudaFree(min_y));
  checkCudaErrors(cudaFree(min_z));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_z));

  delete[] x;
  delete[] y;
  delete[] z;

  return 0;
}