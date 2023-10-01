// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Std C++
#include <stdio.h>
#include <iostream>

// Arc Cuda
#include "ArcCudaMatrixMultiply.h"

__global__ void matrixMultiply(int** ppMatrix1, int** ppMatrix2, int **ppMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Pvalue = 0;

    //int runningResult = 0;
    //for (int k = 0; k < Width; k++)
    //{
    //    float Mdelement = Md[ty * Width + k];
    //    float Ndelement = Nd[k * Width + tx];
    //    // Pd[ty*Width+tx] += Mdelement * Ndelement; – NO!
    //    runningResult += Mdelement * Ndelement;
    //}
    //
    //Pd[ty * Width + tx] = runningResult;

    //for (int rowIndex = 0; rowIndex < matrixSizeM; ++rowIndex)
    //{
    //    for (int columnIndex = 0; columnIndex < matrixSizeP; ++columnIndex)
    //    {
    //        _ppMatrix3[rowIndex][columnIndex] = dotProduct(_ppMatrix1, _ppMatrix2, rowIndex, columnIndex, _matrixSizeN);
    //    }
    //}
    //ppMatrix3[tx][ty] = Pvalue;
}

bool calcMatrixMultiply(int** ppMatrix1, int** ppMatrix2, int** ppMatrix3, const int matrixSizeM, const int matrixSizeN, const int matrixSizeP)
{
    int** ppCudaMatrix1;
    int** ppCudaMatrix2;
    int** ppCudaMatrix3;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Could not set cuda device.\n";
        return false;
    }

    cudaStatus = cudaMalloc((void**)&ppCudaMatrix1, size_t(matrixSizeM) * size_t(matrixSizeN) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Could not allocate the first Cuda Matrix.\n";
        return false;
    }

    cudaStatus = cudaMalloc((void**)&ppCudaMatrix2, size_t(matrixSizeN) * size_t(matrixSizeP) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Could not allocate the second Cuda Matrix.\n";
        return false;
    }

    cudaStatus = cudaMalloc((void**)&ppCudaMatrix3, size_t(matrixSizeM) * size_t(matrixSizeP) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Could not allocate the third Cuda Matrix.\n";
        return false;
    }

    cudaStatus = cudaMemcpy(ppCudaMatrix1, ppMatrix1, size_t(matrixSizeM) * size_t(matrixSizeN) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Could not copy the memory from the host first matrix to the device first Matrix.\n";
        return false;
    }

    cudaStatus = cudaMemcpy(ppCudaMatrix2, ppMatrix2, size_t(matrixSizeN) * size_t(matrixSizeP) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Could not copy the memory from the host second matrix to the device second Matrix.\n";
        return false;
    }

    dim3 blockSize(matrixSizeN, matrixSizeN);
    dim3 gridSize(1, 1);

    matrixMultiply<<<gridSize, blockSize>>>(ppMatrix1, ppMatrix2, ppMatrix3, matrixSizeM, matrixSizeN, matrixSizeP);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Error processing Cuda matrix multiplication.\n";
        return false;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Error processing synchronizing Cuda kernel threads.\n";
        return false;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ppCudaMatrix3, ppMatrix3, size_t(matrixSizeM) * size_t(matrixSizeP) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        std::cout << "Could not copy the memory from the device third matrix to the host third matrix.\n";
        return false;
    }

    cudaFree(ppCudaMatrix1);
    cudaFree(ppCudaMatrix2);
    cudaFree(ppCudaMatrix3);

    return true;
}