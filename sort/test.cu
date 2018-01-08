#include <iostream>
#include <cuda.h>

int main()
{
	int *x = new int[10];

	for (int i = 0; i < 10; i++) {
		x[i] = i + 1;
	}

	int *d_x;

	cudaMalloc((void**)&d_x, sizeof(int)*10);
	cudaMemcpy(d_x, x, sizeof(int)*10, cudaMemcpyHostToDevice);

	int *d_y = d_x + 5;

	int *y = new int[5];
	cudaMemcpy(y, d_y, sizeof(int)*5, cudaMemcpyDeviceToHost);

	std::cout << y[0] << " " << y[1] << " " << y[2] << " " << y[3] << " " << y[4] << std::endl;

	return 0;
}