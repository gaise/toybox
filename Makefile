all: mem_cuda

mem_cuda: mem_cuda.cu
	nvcc -std=c++11 -o mem_cuda mem_cuda.cu
