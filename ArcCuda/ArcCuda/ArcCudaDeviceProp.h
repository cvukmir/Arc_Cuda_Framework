#ifndef ARC_CUDA_DEVICE_PROP_H
#define ARC_CUDA_DEVICE_PROP_H

#include <string>

class ArcCudaDeviceProp
{
public:
	ArcCudaDeviceProp(const int deviceId);
	~ArcCudaDeviceProp();

	virtual std::string toString();

private:

	// Methods
	void printField(std::string outputStringName, std::string outputStringValue);
};
#endif // !ARC_CUDA_DEVICE_PROP_H