#include "ArcCudaDeviceprop.cuh"

#include <cinttypes>

#include <iomanip>
#include <iostream>

static const int FIELD_WIDTH = 45;

ArcCudaDeviceProp::ArcCudaDeviceProp(const int deviceId)
{
	cudaGetDeviceProperties(&_deviceProperties, deviceId);
}

ArcCudaDeviceProp::~ArcCudaDeviceProp()
{
}

std::string ArcCudaDeviceProp::toString()
{
    printField("Name",                    std::string(_deviceProperties.name));
    printField("UUID",                    std::string(_deviceProperties.uuid.bytes));
    printField("LUID",                    std::string(_deviceProperties.luid));
    printField("LUID Device Node Mask",   std::to_string(_deviceProperties.luidDeviceNodeMask));
    printField("Total Global Memory",     std::to_string(int64_t(_deviceProperties.totalGlobalMem)));
    printField("Shared Memory Per Block", std::to_string(_deviceProperties.sharedMemPerBlock));
    printField("Regs Per Block",          std::to_string(_deviceProperties.regsPerBlock));
    printField("Warp Size",               std::to_string(_deviceProperties.warpSize));
    printField("Mem Pitch",               std::to_string(_deviceProperties.memPitch));
    printField("Max Threads Per Block",   std::to_string(_deviceProperties.maxThreadsPerBlock));

    for (uint16_t maxThreadsDimIndex = 0U; maxThreadsDimIndex < 3U; ++maxThreadsDimIndex)
    {
        printField("Max Threads Dim", std::to_string(_deviceProperties.maxThreadsDim[maxThreadsDimIndex]));
    }

    for (uint16_t maxGridSizeIndex = 0U; maxGridSizeIndex < 3U; ++maxGridSizeIndex)
    {
        printField("Max Grid Size", std::to_string(_deviceProperties.maxGridSize[maxGridSizeIndex]));
    }

    printField("Clock Rate",                 std::to_string(_deviceProperties.clockRate));
    printField("Total Const Mem",            std::to_string(_deviceProperties.totalConstMem));
    printField("Major",                      std::to_string(_deviceProperties.major));
    printField("Minor",                      std::to_string(_deviceProperties.minor));
    printField("Texture Alignment",          std::to_string(_deviceProperties.textureAlignment));
    printField("Texture Pitch Alignment",    std::to_string(_deviceProperties.texturePitchAlignment));
    printField("Device Overlap",             std::to_string(_deviceProperties.deviceOverlap));
    printField("Multi Procesor Count",       std::to_string(_deviceProperties.multiProcessorCount));
    printField("Kernel Exec Timout Enabled", std::to_string(_deviceProperties.kernelExecTimeoutEnabled));
    printField("Integrated",                 std::to_string(_deviceProperties.integrated));
    printField("Can Map Host Memory",        std::to_string(_deviceProperties.canMapHostMemory));
    printField("Compute Mode",               std::to_string(_deviceProperties.computeMode));
    printField("Max Texture 1D",             std::to_string(_deviceProperties.maxTexture1D));
    printField("Max Texture 1D Mipmap",      std::to_string(_deviceProperties.maxTexture1DMipmap));
    printField("Max Texture 1D Linear",      std::to_string(_deviceProperties.maxTexture1DLinear));

    for (uint16_t maxTexture2DIndex = 0U; maxTexture2DIndex < 2U; ++maxTexture2DIndex)
    {
        printField("Max Texture 2D[" + std::to_string(maxTexture2DIndex) + "]", std::to_string(_deviceProperties.maxTexture2D[maxTexture2DIndex]));
    }

    for (uint16_t maxTexture2DMipmapIndex = 0U; maxTexture2DMipmapIndex < 2U; ++maxTexture2DMipmapIndex)
    {
        printField("Max Texture 2D Mipmap[" + std::to_string(maxTexture2DMipmapIndex) + "]", std::to_string(_deviceProperties.maxTexture2DMipmap[maxTexture2DMipmapIndex]));
    }

    for (uint16_t maxTexture2DLinearIndex = 0U; maxTexture2DLinearIndex < 3U; ++maxTexture2DLinearIndex)
    {
        printField("Max Texture 2D Linear[" + std::to_string(maxTexture2DLinearIndex) + "]", std::to_string(_deviceProperties.maxTexture2DLinear[maxTexture2DLinearIndex]));
    }

    for (uint16_t maxTexture2DGatherIndex = 0U; maxTexture2DGatherIndex < 2U; ++maxTexture2DGatherIndex)
    {
        printField("Max Texture 2D Gather[" + std::to_string(maxTexture2DGatherIndex) + "]", std::to_string(_deviceProperties.maxTexture2DGather[maxTexture2DGatherIndex]));
    }

    for (uint16_t maxTexture3DIndex = 0U; maxTexture3DIndex < 3U; ++maxTexture3DIndex)
    {
        printField("Max Texture 3D[" + std::to_string(maxTexture3DIndex) + "]", std::to_string(_deviceProperties.maxTexture3D[maxTexture3DIndex]));
    }

    for (uint16_t maxTexture3DAltIndex = 0U; maxTexture3DAltIndex < 3U; ++maxTexture3DAltIndex)
    {
        printField("Max Texture 3D[" + std::to_string(maxTexture3DAltIndex) + "]", std::to_string(_deviceProperties.maxTexture3DAlt[maxTexture3DAltIndex]));
    }

    printField("Max Texture Cubemap", std::to_string(_deviceProperties.maxTextureCubemap));

    for (uint16_t maxTexture1DLayeredIndex = 0U; maxTexture1DLayeredIndex < 2U; ++maxTexture1DLayeredIndex)
    {
        printField("Max Texture 1D Layered[" + std::to_string(maxTexture1DLayeredIndex) + "]", std::to_string(_deviceProperties.maxTexture1DLayered[maxTexture1DLayeredIndex]));
    }

    for (uint16_t maxTexture2DLayeredIndex = 0U; maxTexture2DLayeredIndex < 3U; ++maxTexture2DLayeredIndex)
    {
        printField("Max Texture 2D Layered[" + std::to_string(maxTexture2DLayeredIndex) + "]", std::to_string(_deviceProperties.maxTexture2DLayered[maxTexture2DLayeredIndex]));
    }

    for (uint16_t maxTextureCubemapLayeredIndex = 0U; maxTextureCubemapLayeredIndex < 2U; ++maxTextureCubemapLayeredIndex)
    {
        printField("Max Texture Cubemap Layered[" + std::to_string(maxTextureCubemapLayeredIndex) + "]", std::to_string(_deviceProperties.maxTextureCubemapLayered[maxTextureCubemapLayeredIndex]));
    }

    printField("Max Surface 1D", std::to_string(_deviceProperties.maxSurface1D));

    for (uint16_t maxSurface2DIndex = 0U; maxSurface2DIndex < 2U; ++maxSurface2DIndex)
    {
        printField("Max Surface 2D[" + std::to_string(maxSurface2DIndex) + "]", std::to_string(_deviceProperties.maxSurface2D[maxSurface2DIndex]));
    }

    for (uint16_t maxSurface3DIndex = 0U; maxSurface3DIndex < 3U; ++maxSurface3DIndex)
    {
        printField("Max Surface 3D[" + std::to_string(maxSurface3DIndex) + "]", std::to_string(_deviceProperties.maxSurface3D[maxSurface3DIndex]));
    }

    for (uint16_t maxSurface1DLayeredIndex = 0U; maxSurface1DLayeredIndex < 2U; ++maxSurface1DLayeredIndex)
    {
        printField("Max Surface 1D Layered[" + std::to_string(maxSurface1DLayeredIndex) + "]", std::to_string(_deviceProperties.maxSurface1DLayered[maxSurface1DLayeredIndex]));
    }

    for (uint16_t maxSurface2DLayeredIndex = 0U; maxSurface2DLayeredIndex < 3U; ++maxSurface2DLayeredIndex)
    {
        printField("Max Surface 2D Layered[" + std::to_string(maxSurface2DLayeredIndex) + "]", std::to_string(_deviceProperties.maxSurface2DLayered[maxSurface2DLayeredIndex]));
    }

    printField("Max Surface Cubemap", std::to_string(_deviceProperties.maxSurfaceCubemap));

    for (uint16_t maxSurfaceCubemapLayeredIndex = 0U; maxSurfaceCubemapLayeredIndex < 2U; ++maxSurfaceCubemapLayeredIndex)
    {
        printField("Max Surface Cubemap Layered[" + std::to_string(maxSurfaceCubemapLayeredIndex) + "]", std::to_string(_deviceProperties.maxSurface2DLayered[maxSurfaceCubemapLayeredIndex]));
    }

    printField("Surface Alignment",                            std::to_string(_deviceProperties.surfaceAlignment));
    printField("Concurent Kernels",                            std::to_string(_deviceProperties.concurrentKernels));
    printField("ECC Enabled",                                  std::to_string(_deviceProperties.ECCEnabled));
    printField("PCI Bus ID",                                   std::to_string(_deviceProperties.pciBusID));
    printField("PCI Device ID",                                std::to_string(_deviceProperties.pciDeviceID));
    printField("PCI Domain ID",                                std::to_string(_deviceProperties.pciDomainID));
    printField("TCC Driver",                                   std::to_string(_deviceProperties.tccDriver));
    printField("Async Engine Count",                           std::to_string(_deviceProperties.asyncEngineCount));
    printField("Unified Addressing",                           std::to_string(_deviceProperties.unifiedAddressing));
    printField("Memory Clock Rate",                            std::to_string(_deviceProperties.memoryClockRate));
    printField("Memory Bus Width",                             std::to_string(_deviceProperties.memoryBusWidth));
    printField("L2 Cache Size",                                std::to_string(_deviceProperties.l2CacheSize));
    printField("Persisting L2 Cache Max Size",                 std::to_string(_deviceProperties.persistingL2CacheMaxSize));
    printField("Max Threads Per Multiprocessor",               std::to_string(_deviceProperties.maxThreadsPerMultiProcessor));
    printField("Strea Priorities Suported",                    std::to_string(_deviceProperties.streamPrioritiesSupported));
    printField("Global L1 Cache Supported",                    std::to_string(_deviceProperties.globalL1CacheSupported));
    printField("Local L2 Cache Supported",                     std::to_string(_deviceProperties.localL1CacheSupported));
    printField("Shared Mem Per Multiprocessor",                std::to_string(_deviceProperties.sharedMemPerMultiprocessor));
    printField("Regs Per Multiprocessor",                      std::to_string(_deviceProperties.regsPerMultiprocessor));
    printField("Is Multi GPU Board",                           std::to_string(_deviceProperties.isMultiGpuBoard));
    printField("Multi GPU Board Group ID",                     std::to_string(_deviceProperties.multiGpuBoardGroupID));
    printField("Host Native Atomic Supported",                 std::to_string(_deviceProperties.hostNativeAtomicSupported));
    printField("Single To Double Precision Perf Ratio",        std::to_string(_deviceProperties.singleToDoublePrecisionPerfRatio));
    printField("Pageable Memory Access",                       std::to_string(_deviceProperties.pageableMemoryAccess));
    printField("Concurrent Managed Access",                    std::to_string(_deviceProperties.concurrentManagedAccess));
    printField("Compute Preemption Supported",                 std::to_string(_deviceProperties.computePreemptionSupported));
    printField("Can Use Host Pointer For Registered Mem",      std::to_string(_deviceProperties.canUseHostPointerForRegisteredMem));
    printField("Cooperative Launch",                           std::to_string(_deviceProperties.cooperativeLaunch));
    printField("Cooperative Multi Device Launch",              std::to_string(_deviceProperties.cooperativeMultiDeviceLaunch));
    printField("Shared Mem per Block Option",                  std::to_string(_deviceProperties.sharedMemPerBlockOptin));
    printField("Pageable Memory Access Uses Host Page Tables", std::to_string(_deviceProperties.pageableMemoryAccessUsesHostPageTables));
    printField("Direct Managed Mem Access From Host",          std::to_string(_deviceProperties.directManagedMemAccessFromHost));
    printField("Max Blocks Per Multi Processor",               std::to_string(_deviceProperties.maxBlocksPerMultiProcessor));
    printField("Reserved Shared Mem Per Block",                std::to_string(_deviceProperties.reservedSharedMemPerBlock));
    printField("Host Register Supported",                      std::to_string(_deviceProperties.hostRegisterSupported));
    printField("Sparse Cuda Array Supported",                  std::to_string(_deviceProperties.sparseCudaArraySupported));
    printField("Host Register Read Only Supported",            std::to_string(_deviceProperties.hostRegisterReadOnlySupported));
    printField("Timeline Semaphore Interop Supported",         std::to_string(_deviceProperties.timelineSemaphoreInteropSupported));
    printField("Memory Pools Supported",                       std::to_string(_deviceProperties.memoryPoolsSupported));
    printField("GPU Direct RDMA Supported",                    std::to_string(_deviceProperties.gpuDirectRDMASupported));
    printField("GPU Direct RDMA Flush Writes Options",         std::to_string(_deviceProperties.gpuDirectRDMAFlushWritesOptions));
    printField("GPU Direct RDMA Writes Ordering",              std::to_string(_deviceProperties.gpuDirectRDMAWritesOrdering));
    printField("Memory Pool Supported Handle Types",           std::to_string(_deviceProperties.memoryPoolSupportedHandleTypes));
    printField("Deffered Mapping Cuda Array Supported",        std::to_string(_deviceProperties.deferredMappingCudaArraySupported));
    printField("IPC Event Supported",                          std::to_string(_deviceProperties.ipcEventSupported));
    printField("Cluster Launch",                               std::to_string(_deviceProperties.clusterLaunch));
    printField("Unified Function Pointers",                    std::to_string(_deviceProperties.unifiedFunctionPointers));

    for (uint16_t reserved2Index = 0U; reserved2Index < 2U; ++reserved2Index)
    {
        printField("Reserved2[" + std::to_string(reserved2Index) + "]", std::to_string(_deviceProperties.reserved2[reserved2Index]));
    }

    for (uint16_t reservedIndex = 0U; reservedIndex < 61U; ++reservedIndex)
    {
        printField("Reserved[" + std::to_string(reservedIndex) + "]", std::to_string(_deviceProperties.reserved[reservedIndex]));
    }

    return std::string();
}

// Private Methods

void ArcCudaDeviceProp::printField(std::string outputStringName, std::string outputStringValue)
{
    std::cout << std::setfill('_') << std::setw(FIELD_WIDTH) << std::left << outputStringName << "|" << outputStringValue << '\n';
}