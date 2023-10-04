// Std C++
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

// ArcMain
#include "ArcAssignment2.h"

// Arc Cuda
#pragma comment(lib, "ArcCuda.lib")
#include "ArcCudaMatrixMultiply.h"

// Constants //

const std::string OUTPUT_LINE     = std::string(25, '*') + std::string("\n");
const int         MAX_ARRAY_SIZE  = 10;
const int         MIN_ARRAY_SIZE  = 3;
const int         MAX_ARRAY_VALUE = 100.0f;
const int         TEST_SIZE       = 32;
const size_t      FLOAT_SIZE      = sizeof(float);

#define CALC_COLUMN_OFFSET(column) (FLOAT_SIZE * column)
#define CALC_RANDOM_FLOAT(randNum, max, min) (min + (static_cast<float>(randNum) / static_cast<float>(RAND_MAX / (max - min)))) // ??? (float(rand()) / float(RAND_MAX)) * value_width + min_value
#define CALC_RANDOM_INT(randNum, max, min) (randNum % max + min)

// Constructors //

ArcAssignment2::ArcAssignment2() :
	_pMatrix1(nullptr),
	_pMatrix2(nullptr),
	_pMatrix3(nullptr),
	_matrixSizeM(0),
	_matrixSizeN(0),
	_matrixSizeP(0)
{
}

ArcAssignment2::~ArcAssignment2()
{
	if (_pMatrix1 != nullptr)
	{
		delete[] _pMatrix1;
	}

	if (_pMatrix2 != nullptr)
	{
		delete[] _pMatrix2;
	}

	if (_pMatrix3 != nullptr)
	{
		if (_pMatrix3 != nullptr)
		{
			delete[] _pMatrix3;
		}
	}
}

// Public Methods //

bool ArcAssignment2::runAssignment2()
{
	generateMatrices();

	std::cout << "Matrix 1 Size: [" << _matrixSizeM << "," << _matrixSizeN << "]\n";
	std::cout << "Matrix 2 Size: [" << _matrixSizeN << "," << _matrixSizeP << "]\n";
	std::cout << "Matrix 3 Size: [" << _matrixSizeM << "," << _matrixSizeP << "]\n";

	std::cout << "Printing Matrix 1" << '\n';

	printMatrix(_pMatrix1, _matrixSizeM, _matrixSizeN);

	std::cout << "Printing Matrix 2" << '\n';

	printMatrix(_pMatrix2, _matrixSizeN, _matrixSizeP);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	multiplyMatricesCPU();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Printing Matrix 3 - Generated From CPU in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanoseconds." << '\n';

	printMatrix(_pMatrix3, _matrixSizeM, _matrixSizeP);

	begin = std::chrono::steady_clock::now();

	for (int rowIndex = 0; rowIndex < _matrixSizeM; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _matrixSizeP; ++columnIndex)
		{
			_pMatrix3[rowIndex * _matrixSizeP + columnIndex] = 0.0;
		}
	}

	if (!multiplyMatricesGPU())
	{
		return false;
	}

	end = std::chrono::steady_clock::now();

	std::cout << "Printing Matrix 3 - Generated From GPU in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanoseconds." << '\n';

	printMatrix(_pMatrix3, _matrixSizeM, _matrixSizeP);

	return true;
}

// Private Methods // 

float ArcAssignment2::dotProduct(float* pMatrix1, float* pMatrix2, const int rowIndex, const int columnIndex, const int size)
{
	float runningTotal = 0;

	for (int i = 0; i < size; ++i)
	{
		runningTotal += pMatrix1[rowIndex * size + i] * pMatrix2[i * size + columnIndex];
	}

	return runningTotal;
}

void ArcAssignment2::fillMatrix(float* pMatrix, const int numberOfRows, const int numberOfColumns)
{
	srand(unsigned int(time(NULL)));

	// Change this to a single loop based on pointers.

	for (int rowIndex = 0; rowIndex < numberOfRows; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < numberOfColumns; ++columnIndex)
		{
			pMatrix[rowIndex * numberOfColumns + columnIndex] = (rand() / static_cast<float>(RAND_MAX)) * MAX_ARRAY_VALUE;
		}
	}
}

void ArcAssignment2::generateMatrices()
{
	initializeSizes();
	initializeMatrices();

	for (int rowIndex = 0; rowIndex < _matrixSizeM; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _matrixSizeN; ++columnIndex)
		{
			_pMatrix1[rowIndex * _matrixSizeN + columnIndex] = 0.0;
		}
	}
	
	for (int rowIndex = 0; rowIndex < _matrixSizeN; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _matrixSizeP; ++columnIndex)
		{
			_pMatrix2[rowIndex * _matrixSizeP + columnIndex] = 0.0;
		}
	}

	fillMatrix(_pMatrix1, _matrixSizeM, _matrixSizeN);
	fillMatrix(_pMatrix2, _matrixSizeN, _matrixSizeP);
}

void ArcAssignment2::initializeMatrices()
{
	_pMatrix1 = new float[_matrixSizeM * _matrixSizeN];
	_pMatrix2 = new float[_matrixSizeN * _matrixSizeP];
	_pMatrix3 = new float[_matrixSizeM * _matrixSizeP];
}

void ArcAssignment2::initializeSizes()
{
	srand(unsigned int(time(NULL)));

	_matrixSizeM = TEST_SIZE;
	_matrixSizeN = TEST_SIZE;
	_matrixSizeP = TEST_SIZE;
}

void ArcAssignment2::multiplyMatricesCPU()
{
	for (int rowIndex = 0; rowIndex < _matrixSizeM; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _matrixSizeP; ++columnIndex)
		{
			_pMatrix3[rowIndex * _matrixSizeP + columnIndex] = dotProduct(_pMatrix1, _pMatrix2, rowIndex, columnIndex, _matrixSizeN);
		}
	}
}

bool ArcAssignment2::multiplyMatricesGPU()
{
	return calcMatrixMultiply(_pMatrix1, _pMatrix2, _pMatrix3, _matrixSizeM, _matrixSizeN, _matrixSizeP);
}

void ArcAssignment2::printMatrix(float* pMatrix, const int numberOfRows, const int numberOfColumns)
{
	std::cout << OUTPUT_LINE;

	for (int rowIndex = 0; rowIndex < numberOfRows; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < numberOfColumns; ++columnIndex)
		{
			std::cout << '|' << std::setfill(' ') << std::setw(4) << std::setprecision(7) << pMatrix[rowIndex * numberOfColumns + columnIndex];
		}

		std::cout << '\n';
	}

	std::cout << OUTPUT_LINE;
}
