#include <iostream>
#include <cuda.h>

#define HANDLE_ERROR(x) checkCudaError(x, __LINE__)

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
	cudaDeviceProp prop;

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++)
		{
			HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
			std::cout << "   --- General Information for device " << i << " ---" << std::endl;
			std::cout << "Name: " << prop.name << std::endl;
			std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
			std::cout << "Clock rate: " << prop.clockRate << std::endl;
			std::cout << "Device copy overlap: ";

			if (prop.deviceOverlap)
				std::cout << "Enabled" << std::endl;
			else
				std::cout << "Disabled" << std::endl;

			std::cout << "Kernel execition timeout: ";

			if (prop.kernelExecTimeoutEnabled)
				std::cout << "Enabled" << std::endl;
			else
				std::cout << "Disabled" << std::endl;

			std::cout << "GPU integrated: " << prop.integrated << std::endl;

			std::cout << std::endl;
			std::cout << "   --- Memory Information for device " << i << " ---"<< std::endl;
			std::cout << "Total global mem: " << prop.totalGlobalMem << std::endl;
			std::cout << "Total constant mem: " << prop.totalConstMem << std::endl;
			std::cout << "Max mem pitch: " << prop.memPitch << std::endl;
			std::cout << "Texture Alignment: " << prop.textureAlignment << std::endl;

			std::cout << std::endl;

			std::cout << "   --- MP Information for device " << i << " ---" << std::endl;
			std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
			std::cout << "Shared mem per mp: " << prop.sharedMemPerBlock << std::endl;
			std::cout << "Registers per mp: " << prop.regsPerBlock << std::endl;
			std::cout << "Threads in warp: " << prop.warpSize << std::endl;
			std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
			std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", "
								<< prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
			std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", "
								<< prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

			std::cout << std::endl;
			std::cout << "   ----------------------------------   " << std::endl << std::endl;
		}

	return 0;
}