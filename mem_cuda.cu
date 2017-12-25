#include <iostream>
#include <cuda.h>
#include <chrono>

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
  double array[20];
  
  for (i = 1; i <= 20; i+=1) {
    s = (float *)malloc(sizeof(float)*i*100);
    checkCudaError(cudaMalloc((void**)&dev_s, sizeof(float)*i*100), __LINE__);
    
    start = std::chrono::system_clock::now();

    checkCudaError(cudaMemcpy(dev_s, s, sizeof(float)*i*100, cudaMemcpyHostToDevice), __LINE__);

    checkCudaError(cudaThreadSynchronize(), __LINE__);

    end = std::chrono::system_clock::now();

    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    array[(i/5)-1] = time;
  }

  for (i = 0; i < 20; i++) {
    std::cout << (i+1)*100 << " float : " << array[i] << "sec." << std::endl;
  }

  return 0;
} 
