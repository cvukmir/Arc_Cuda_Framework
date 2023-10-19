// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Std C++
#include <string>


class ArcCudaDeviceProp : cudaDeviceProp
{
public:
	ArcCudaDeviceProp(const int deviceId);
	~ArcCudaDeviceProp();

	std::string toString();

private:
	
	// Methods
	void printField(std::string outputStringName, std::string outputStringValue);

	// Variables

	cudaDeviceProp _deviceProperties;
};