// Std C++
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

// ArcMain
#include "ArcAssignment2.h"

// Arc Cuda
#include "ArcCudaMatrixMultiply.h"

// Constants //

const std::string OUTPUT_LINE     = std::string(25, '*') + std::string("\n");
const int         MAX_ARRAY_SIZE  = 10;
const int         MIN_ARRAY_SIZE  = 3;
const int         MAX_ARRAY_VALUE = 100;

// Constructors //

ArcAssignment2::ArcAssignment2() :
	_ppMatrix1(nullptr),
	_ppMatrix2(nullptr),
	_ppMatrix3(nullptr),
	_matrixSizeM(0),
	_matrixSizeN(0),
	_matrixSizeP(0)
{
}

ArcAssignment2::~ArcAssignment2()
{
	if (_ppMatrix1 != nullptr)
	{
		for (int i = 0; i < _matrixSizeM; ++i)
		{
			delete[] _ppMatrix1[i];
		}
		delete[] _ppMatrix1;
	}

	if (_ppMatrix2 != nullptr)
	{
		for (int i = 0; i < _matrixSizeN; ++i)
		{
			delete[] _ppMatrix2[i];
		}
		delete[] _ppMatrix2;
	}

	if (_ppMatrix3 != nullptr)
	{
		if (_ppMatrix3 != nullptr)
		{
			for (int i = 0; i < _matrixSizeM; ++i)
			{
				delete[] _ppMatrix3[i];
			}
			delete[] _ppMatrix3;
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

	printMatrix(_ppMatrix1, _matrixSizeM, _matrixSizeN);

	std::cout << "Printing Matrix 2" << '\n';

	printMatrix(_ppMatrix2, _matrixSizeN, _matrixSizeP);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	multiplyMatricesCPU();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Printing Matrix 3 - Generated From CPU in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanoseconds." << '\n';

	printMatrix(_ppMatrix3, _matrixSizeM, _matrixSizeP);

	begin = std::chrono::steady_clock::now();

	multiplyMatricesGPU();

	end = std::chrono::steady_clock::now();

	std::cout << "Printing Matrix 3 - Generated From GPU in " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanoseconds." << '\n';

	printMatrix(_ppMatrix3, _matrixSizeM, _matrixSizeP);

	return true;
}

// Private Methods // 

int ArcAssignment2::dotProduct(int** ppMatrix1, int** ppMatrix2, const int rowIndex, const int columnIndex, const int size)
{
	int runningTotal = 0;

	for (int i = 0; i < size; ++i)
	{
		runningTotal += ppMatrix1[rowIndex][i] * ppMatrix2[i][columnIndex];
	}

	return runningTotal;
}

// TODO: FIX
void ArcAssignment2::fillMatrix(int** ppMatrix, const int numberOfRows, const int numberOfColumns)
{
	srand(unsigned int(time(NULL)));

	ppMatrix = new int*[numberOfRows];
	for (int i = 0; i < numberOfRows; ++i)
	{
		ppMatrix[i] = new int[numberOfColumns];

		for (int j = 0; j < numberOfColumns; ++j)
		{
			ppMatrix[i][j] = rand() % 10;
		}
	}
}

void ArcAssignment2::generateMatrices()
{
	srand(unsigned int(time(NULL)));

	_matrixSizeM = rand() % MAX_ARRAY_SIZE + MIN_ARRAY_SIZE;
	_matrixSizeN = rand() % MAX_ARRAY_SIZE + MIN_ARRAY_SIZE;
	_matrixSizeP = rand() % MAX_ARRAY_SIZE + MIN_ARRAY_SIZE;
	// Compiler (flat(rand()) / float(RAND_MAX)) * value_width + min_valut
	

	_ppMatrix1 = new int* [_matrixSizeM];
	for (int i = 0; i < _matrixSizeM; ++i)
	{
		_ppMatrix1[i] = new int[_matrixSizeN];

		for (int j = 0; j < _matrixSizeN; ++j)
		{
			_ppMatrix1[i][j] = rand() % MAX_ARRAY_VALUE;
		}
	}

	_ppMatrix2 = new int* [_matrixSizeN];
	for (int i = 0; i < _matrixSizeN; ++i)
	{
		_ppMatrix2[i] = new int[_matrixSizeP];

		for (int j = 0; j < _matrixSizeP; ++j)
		{
			_ppMatrix2[i][j] = rand() % MAX_ARRAY_VALUE;
		}
	}

	//fillMatricies(_ppMatrix1, _matrixSizeM, _matrixSizeN);
	//fillMatricies(_ppMatrix2, _matrixSizeN, _matrixSizeP);

	// Initialize the combination matrix
	_ppMatrix3 = new int* [_matrixSizeM];
	for (int i = 0; i < _matrixSizeM; ++i)
	{
		_ppMatrix3[i] = new int[_matrixSizeP];
	}
}

void ArcAssignment2::multiplyMatricesCPU()
{
	for (int rowIndex = 0; rowIndex < _matrixSizeM; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < _matrixSizeP; ++columnIndex)
		{
			_ppMatrix3[rowIndex][columnIndex] = dotProduct(_ppMatrix1, _ppMatrix2, rowIndex, columnIndex, _matrixSizeN);
		}
	}
}

bool ArcAssignment2::multiplyMatricesGPU()
{
	calcMatrixMultiply(_ppMatrix1, _ppMatrix2, _ppMatrix3, _matrixSizeM, _matrixSizeN, _matrixSizeP);

	return true;
}

void ArcAssignment2::printMatrix(int** ppMatrix, const int numberOfRows, const int numberOfColumns)
{
	std::cout << OUTPUT_LINE;

	for (int rowIndex = 0; rowIndex < numberOfRows; ++rowIndex)
	{
		for (int columnIndex = 0; columnIndex < numberOfColumns; ++columnIndex)
		{
			std::cout << '|' << std::setfill(' ') << std::setw(4) << ppMatrix[rowIndex][columnIndex];
		}

		std::cout << '\n';
	}

	std::cout << OUTPUT_LINE;
}
