#include <iostream>
#include <cuda.h>
#include <random>

#define N 4096
#define THREAD 256
#define BLOCK 18

#define HANDLE_ERROR(x) checkCudaError(x, __LINE__)

void checkCudaError(cudaError_t msg, int x)
{
  if (msg != cudaSuccess) {
    fprintf(stderr, "line: %d %s\n", x, cudaGetErrorString(msg));
    exit(1);
  }
  return;
}

__global__ void kernel(double *x, double *y, double *z, double *tf, double *ox, double *oy, double *oz)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < N; i+=stride)
		{
			ox[i] = x[i] * tf[0] + y[i] * tf[1] + z[i] * tf[2] + tf[3];
			oy[i] = x[i] * tf[4] + y[i] * tf[5] + z[i] * tf[6] + tf[7];
			oz[i] = x[i] * tf[8] + y[i] * tf[9] + z[i] * tf[10] + tf[11];
		}
}

int main()
{
	cudaEvent_t start[2], stop[2];
	float time;

	double org_x[N], org_y[N], org_z[N];

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<double> rand100(0.0, 100.0);

	for (int i = 0; i < N; i++)
		{
			org_x[i] = rand100(mt);
			org_y[i] = rand100(mt);
			org_z[i] = rand100(mt);
		}
	
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));

	double tf_org[12] = {1.0, 1.5, 1.2, 11.0,
									 0.9, 1.1, 0.8, 2.0,
											 1.0, 1.0, 0.6, -1.0};
	double *tf;

	HANDLE_ERROR(cudaMalloc((void**)&tf, sizeof(double)*12));
	HANDLE_ERROR(cudaMemcpy(tf, tf_org, sizeof(double)*12, cudaMemcpyHostToDevice));

	double *ox, *oy, *oz;
	HANDLE_ERROR(cudaMalloc((void**)&ox, sizeof(double)*N));
	HANDLE_ERROR(cudaMalloc((void**)&oy, sizeof(double)*N));
	HANDLE_ERROR(cudaMalloc((void**)&oz, sizeof(double)*N));

	double *x, *y, *z;

	for (int i = 0; i < count; i++)
		{
			HANDLE_ERROR(cudaEventCreate(&start[i]));
			HANDLE_ERROR(cudaEventCreate(&stop[i]));

			HANDLE_ERROR(cudaSetDevice(i));

			HANDLE_ERROR(cudaEventRecord(start[i], (cudaStream_t)i));
			
			HANDLE_ERROR(cudaMalloc((void**)&x, sizeof(double)*N));
			HANDLE_ERROR(cudaMalloc((void**)&y, sizeof(double)*N));			
			HANDLE_ERROR(cudaMalloc((void**)&z, sizeof(double)*N));

			HANDLE_ERROR(cudaMemcpy(x, org_x, sizeof(double)*N, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(y, org_y, sizeof(double)*N, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(z, org_z, sizeof(double)*N, cudaMemcpyHostToDevice));

			kernel<<<BLOCK, THREAD>>>(x, y, z, tf, ox, oy, oz);

			HANDLE_ERROR(cudaEventRecord(stop[i], (cudaStream_t)i));
			HANDLE_ERROR(cudaEventSynchronize(stop[i]));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start[i], stop[i]));

			std::cout << "device: " << i << std::endl;
			std::cout << "time: " << time << std::endl;

			HANDLE_ERROR(cudaFree(x));
			HANDLE_ERROR(cudaFree(y));
			HANDLE_ERROR(cudaFree(z));

			HANDLE_ERROR(cudaEventDestroy(start[i]));
			HANDLE_ERROR(cudaEventDestroy(stop[i]));

		}

	HANDLE_ERROR(cudaFree(tf));
	HANDLE_ERROR(cudaFree(ox));
	HANDLE_ERROR(cudaFree(oy));
	HANDLE_ERROR(cudaFree(oz));

	return 0;

}