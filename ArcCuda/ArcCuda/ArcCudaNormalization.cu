// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Std C++
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <chrono>

// ArcCuda
#include "ArcCudaNormalization.h"

const int BLOCK_WIDTH = 1024; // AKA: TILE_WIDTH

__global__ void normalizationKernel(float* pInputArray, int size, float* normalizedValue)
{
	__shared__ float partialSum[BLOCK_WIDTH];

	unsigned int threadX = threadIdx.x;
	unsigned int blockX  = blockIdx.x;

	for (int blockIndex = 0; blockIndex <= size / BLOCK_WIDTH; ++blockIndex)
	{
		partialSum[threadX + (blockX * BLOCK_WIDTH)] = pInputArray[threadX + (blockX * BLOCK_WIDTH)] * pInputArray[threadX + (blockX * BLOCK_WIDTH)];

		for (unsigned int stride = BLOCK_WIDTH >> 1; stride > 0; stride >>= 1)
		{
			__syncthreads();
			if (threadX < stride)
			{
				partialSum[threadX] += partialSum[threadX + stride];
			}
		}

		__syncthreads();
	}

	if (threadX + (blockX * BLOCK_WIDTH) >= size)
	{
		return;
	}

	(*normalizedValue) += partialSum[0];

	if (threadX != 0 && blockX != 0)
	{
		return;
	}

	(*normalizedValue) = std::sqrtf(*normalizedValue);
}


bool calcNormalization(float* pArray, const int size, float* normalizedValue)
{
	float* pCudaArray;
	float* cudaNormalizedValue;

	cudaError_t cudaStatus;

	// Set the device //

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		std::cout << "Could not set Cuda device.\n";
		return false;
	}

	// Allocate the arrays //

	cudaStatus = cudaMalloc((void**)&pCudaArray, size_t(size) * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Could not allocate the first Cuda array.\n";
		return false;
	}

	cudaStatus = cudaMalloc((void**)&cudaNormalizedValue, sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "Could not allocate the first Cuda array.\n";
		return false;
	}

	// Copy the memory from CPU to GPU //

	cudaStatus = cudaMemcpy(pCudaArray, pArray, size_t(size) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		cudaFree(pCudaArray);
		std::cout << "Could not copy the memory from the host first array to the device first array.\n";
		return false;
	}

	// Perform the normalization //

	dim3 threadsPerBlock(BLOCK_WIDTH);
	dim3 numBlocks(ceil(size / float(BLOCK_WIDTH)));
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	normalizationKernel<<<numBlocks, threadsPerBlock>>>(pCudaArray, size, cudaNormalizedValue);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Normalization - Generated From GPU in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanoseconds." << '\n';

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		cudaFree(pCudaArray);
		std::cout << "Error processing Cuda array normalization.\n";
		return false;
	}

	// Copy the memory from the GPU to the CPU //
	
	cudaStatus = cudaMemcpy(normalizedValue, cudaNormalizedValue, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		cudaFree(pCudaArray);
		std::cout << "Could not copy the memory from the device third arra to the host third array.\n";
		return false;
	}

	cudaFree(pCudaArray);

	return true;
}