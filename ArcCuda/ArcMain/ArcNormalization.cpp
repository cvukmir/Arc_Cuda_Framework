// Std C++
#include <random>
#include <math.h>
#include <iostream>

// ArcMain
#include "ArcNormalization.h"

// ArcCuda
#include "ArcCudaNormalization.h"

// Constant(s) //

const float MAX_ARRAY_VALUE = 100.0f;

// Constructor(s) //

ArcNormalization::ArcNormalization()
	: _pArray (nullptr)
	, _normalizedValue (0.00)
	, _size   (0)
{
}

// Destructor(s) //

ArcNormalization::~ArcNormalization()
{
	if (_pArray)
	{
		delete _pArray;
	}
}

// Methods - Public //

bool ArcNormalization::performNormalization()
{
	initializeSize();

	initializeArray();

	fillArray();

	if (!normalizeCpu())
	{
		return false;
	}

	printArray();

	if (!normalizeGpu())
	{
		return false;
	}

	std::cout << "Normalized value: " << _normalizedValue << "\n";

	return true;
}

// Methods - Private //

void ArcNormalization::clearArray()
{
	for (int index = 0; index < _size; ++index)
	{
		_pArray[index] = 0.0;
	}
}

void ArcNormalization::fillArray()
{
	for (int index = 0; index < _size; ++index)
	{
		_pArray[index] = (rand() / static_cast<float>(RAND_MAX)) * MAX_ARRAY_VALUE;
	}
}

void ArcNormalization::initializeArray()
{
	_pArray = new float[_size];

	clearArray();
}

void ArcNormalization::initializeSize()
{
	_size = static_cast<int>(rand() % 100 + 3);
}

bool ArcNormalization::normalizeCpu()
{
	float sumSquaredValue = 0.0;

	for (int index = 0; index < _size; ++index)
	{
		sumSquaredValue += _pArray[index] * _pArray[index];
	}

	_normalizedValue = std::sqrt(sumSquaredValue);

	return true;
}

bool ArcNormalization::normalizeGpu()
{
	return calcNormalization(_pArray, _size, &_normalizedValue);
}

void ArcNormalization::printArray()
{
	std::cout << "Printing the array of size " << _size << ":\n";

	for (int index = 0; index < _size; ++index)
	{
		std::cout << "Index: " << index << " |Value: " << _pArray[index] << "\n";
	}

	std::cout << "\n---------------------\n";
}

