// Cuda Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// C++ Base
#include <string>

class ArcCudaDeviceProp : cudaDeviceProp//ArcCudaDeviceProp, cudaDeviceProp
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