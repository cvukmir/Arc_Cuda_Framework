// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Std C++
#include <stdio.h>
#include <iostream>

// Arc Cuda
#include "ArcCudaMatrixMultiply.h"

__global__ void matrixMultiply(float* pMatrix1, float* pMatrix2, float* pMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    int   threadX = threadIdx.x;
    int   threadY = threadIdx.y;
    float Pvalue  = 0.0f;

    for (int k = 0; k < matrixSizeM; ++k)
    {
        float Mdelement = pMatrix1[threadY * matrixSizeM + k];
        float Ndelement = pMatrix2[k * matrixSizeM + threadX];
        Pvalue += Mdelement * Ndelement;
    }

    pMatrix3[threadY * matrixSizeM + threadX] = Pvalue;
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

    dim3 blockSize(matrixSizeN, matrixSizeN);
    dim3 gridSize(1, 1);

    matrixMultiply<<<gridSize, blockSize>>>(pCudaMatrix1, pCudaMatrix2, pCudaMatrix3, matrixSizeM, matrixSizeN, matrixSizeP);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Error processing Cuda matrix multiplication.\n";
        return false;
    }

    // Synchronize threads //

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        cudaFree(pCudaMatrix1);
        cudaFree(pCudaMatrix2);
        cudaFree(pCudaMatrix3);
        std::cout << "Error processing synchronizing Cuda kernel threads.\n";
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