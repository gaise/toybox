#include <iostream>
#include <cuda.h>
#include <chrono>

#define ITER 200

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
  int i, j;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  double time;
  double array[ITER];
  
  for (i = 1; i <= ITER; i+=1) {
    s = (float *)malloc(sizeof(float)*i*100);
    checkCudaError(cudaMalloc((void**)&dev_s, sizeof(float)*i*100), __LINE__);

    for (j = 0; j < i*100; j++) {
      s[j] = j;
    }
    
    start = std::chrono::system_clock::now();

    checkCudaError(cudaMemcpy(dev_s, s, sizeof(float)*i*100, cudaMemcpyHostToDevice), __LINE__);

    checkCudaError(cudaThreadSynchronize(), __LINE__);

    end = std::chrono::system_clock::now();

    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    array[i-1] = time;
	free(s);
	checkCudaError(cudaFree(dev_s), __LINE__);
  }

  for (i = 0; i < ITER; i++) {
    std::cout << (i+1)*100 << " float : " << array[i] << "sec." << std::endl;
  }

  return 0;
} 
