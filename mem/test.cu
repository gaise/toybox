#include <iostream>
#include <cuda.h>
#include <chrono>

#define SIZE 1000

void checkCudaError(cudaError_t msg, int x)
{
  if (msg != cudaSuccess) {
    fprintf(stderr, "line: %d %s\n", x, cudaGetErrorString(msg));
    exit(1);
  }
  return;
}

int main()
{
  float *s, *dev_s;
  int i;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  double time;
  
  s = (float *)malloc(sizeof(float)*SIZE);
  for (i = 0; i < SIZE; i++) {
    s[i] = i;
  }
  checkCudaError(cudaMalloc((void**)&dev_s, sizeof(float)*SIZE), __LINE__);
  
  start = std::chrono::system_clock::now();

  checkCudaError(cudaMemcpy(dev_s, s, sizeof(float)*SIZE, cudaMemcpyHostToDevice), __LINE__);

  checkCudaError(cudaThreadSynchronize(), __LINE__);

  end = std::chrono::system_clock::now();

  time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  free(s);
  checkCudaError(cudaFree(dev_s), __LINE__);

  std::cout << time << "usec." << std::endl;

  return 0;
} 