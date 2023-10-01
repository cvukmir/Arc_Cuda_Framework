#pragma once

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