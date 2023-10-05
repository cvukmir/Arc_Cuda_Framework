// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Std C++
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdint.h>


// Arc Cuda
#include "ArcCudaMatrixMultiply.h"

const int BLOCK_WIDTH = 32; // AKA: TILE_WIDTH

__global__ void matrixMultiplyDynamic(float* pMatrix1, float* pMatrix2, float* pMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    int   blockX           = blockIdx.x;
    int   blockY           = blockIdx.y;
    int   threadX          = threadIdx.x;
    int   threadY          = threadIdx.y;
    float computedValue    = 0.0f;

    int row = blockY * BLOCK_WIDTH + threadY;
    int col = blockX * BLOCK_WIDTH + threadX;

    if (row >= matrixSizeM || col >= matrixSizeP)
    {
        return;
    }

    for (int k = 0; k < matrixSizeN; ++k)
    {
        computedValue += pMatrix1[row * matrixSizeN + k] * pMatrix2[k * matrixSizeP + col];
    }

    pMatrix3[row * matrixSizeP + col] = computedValue;
}

__global__ void matrixMultiplyTiledBlocksSameSize(float* pMatrix1, float* pMatrix2, float* pMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    //int   threadsPerXBlock = blockDim.x;
    //int   threadsPerYBlock = blockDim.y;
    int   blockX           = blockIdx.x;
    int   blockY           = blockIdx.y;
    int   threadX          = threadIdx.x;
    int   threadY          = threadIdx.y;
    float computedValue    = 0.0f;

    int row = blockY * BLOCK_WIDTH + threadY;
    int col = blockX * BLOCK_WIDTH + threadX;

    for (int k = 0; k < matrixSizeM; ++k)
    {
        computedValue += pMatrix1[row * matrixSizeM + k] * pMatrix2[k * matrixSizeM + col];
    }

    pMatrix3[row * matrixSizeM + col] = computedValue;
}

__global__ void matrixMultiplySameSize(float* pMatrix1, float* pMatrix2, float* pMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    int   threadX          = threadIdx.x;
    int   threadY          = threadIdx.y;
    float computedValue    = 0.0f;

    for (int k = 0; k < matrixSizeM; ++k)
    {
        float matrix1Element = pMatrix1[threadY * matrixSizeM + k];
        float matrix2Element = pMatrix2[k * matrixSizeM + threadX];
        computedValue += matrix1Element * matrix2Element;
    }

    pMatrix3[threadY * matrixSizeM + threadX] = computedValue;
}

bool calcMatrixMultiply(float* pMatrix1, float* pMatrix2, float* pMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    float* pCudaMatrix1;
    float* pCudaMatrix2;
    float* pCudaMatrix3;

    cudaError_t cudaStatus;

    // Set the device //

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Could not set cuda device.\n";
        return false;
    }

    // Allocate the arrays //

    cudaStatus = cudaMalloc((void**)&pCudaMatrix1, size_t(matrixSizeM) * size_t(matrixSizeN) * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Could not allocate the first Cuda Matrix.\n";
        return false;
    }

    cudaStatus = cudaMalloc((void**)&pCudaMatrix2, size_t(matrixSizeN) * size_t(matrixSizeP) * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cudaFree(pCudaMatrix1);
        std::cout << "Could not allocate the second Cuda Matrix.\n";
        return false;
    }

    cudaStatus = cudaMalloc((void**)&pCudaMatrix3, size_t(matrixSizeM) * size_t(matrixSizeP) * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        std::cout << "Could not allocate the third Cuda Matrix.\n";
        return false;
    }

    // Copy the memory from CPU to GPU //

    cudaStatus = cudaMemcpy(pCudaMatrix1, pMatrix1, size_t(matrixSizeM) * size_t(matrixSizeN) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Could not copy the memory from the host first matrix to the device first Matrix.\n";
        return false;
    }

    cudaStatus = cudaMemcpy(pCudaMatrix2, pMatrix2, size_t(matrixSizeN) * size_t(matrixSizeP) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Could not copy the memory from the host second matrix to the device second Matrix.\n";
        return false;
    }

    // Perform the multiplication //
    
    // Same size
    //dim3 blockSize(matrixSizeN, matrixSizeN);
    //dim3 gridSize(1, 1);
    //
    //matrixMultiplySameSize << <gridSize, blockSize >> > (pCudaMatrix1, pCudaMatrix2, pCudaMatrix3, matrixSizeM, matrixSizeN, matrixSizeP);

    // Tiled blocks - same size
    //dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    //dim3 numBlocks(std::ceil(matrixSizeM / static_cast<float>(BLOCK_WIDTH)), std::ceil(matrixSizeM / static_cast<float>(BLOCK_WIDTH)));
    //
    //matrixMultiplyTiledBlocks<<<numBlocks, threadsPerBlock>>>(pCudaMatrix1, pCudaMatrix2, pCudaMatrix3, matrixSizeM, matrixSizeN, matrixSizeP);

    // Dynamic sizes
    // Potential way of solving variable size:
    // x = Number of cols + block_width – 1 / block_width
    // y = Number of rows + block_width – 1 / block_width
    // numRows = sqrt(x) 
    // numCols = sqrt(y) 
    // float decimal;
    // numRows = std::modf()
    // if num is not exactly even, round up
    // dim3 numBlocks (num, num)

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 numBlocks(ceil(matrixSizeP / float(BLOCK_WIDTH)), ceil(matrixSizeM / float(BLOCK_WIDTH)));

    matrixMultiplyDynamic<<<numBlocks, threadsPerBlock>>>(pCudaMatrix1, pCudaMatrix2, pCudaMatrix3, matrixSizeM, matrixSizeN, matrixSizeP);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Error processing Cuda matrix multiplication.\n";
        return false;
    }

    // Copy the memory from the GPU to the CPU //
    
    cudaStatus = cudaMemcpy(pMatrix3, pCudaMatrix3, size_t(matrixSizeM) * size_t(matrixSizeP) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Could not copy the memory from the device third matrix to the host third matrix.\n";
        return false;
    }

    cudaFree(pCudaMatrix1);
    cudaFree(pCudaMatrix2);
    cudaFree(pCudaMatrix3);

    return true;
}

bool isEvenDecimal(float value, float precision)
{
    float remainder = 0.0;
    modf(value, &remainder);

    return fabs((value + remainder) - value) < precision;
}