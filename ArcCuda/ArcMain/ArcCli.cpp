// ArcMain
#include "ArcCli.h"
#include "ArcMatrixMultiply.h"
#include "ArcNormalization.h"

// Std C++
#include <iostream>
#include <iomanip>

// Static Met

bool ArcCli::runInterface()
{
	bool keepRunning = true;
	char option      = INVALID_OPTION;

	while (true)
	{
		std::cout << "Please select an option: ";

		std::cin >> option;

		switch (option)
		{
		case (INVALID_OPTION):
			std::cout << "\n---Invalid Option---\n";
			break;
		case (MATRIX_MULTIPLY_OPTION):
			std::cout << "\n---Running matrix multiplication---\n";
			runMatrixMultiplication();
			break;
		case (NORMALIZER_OPTION):
			std::cout << "\n---Running normalizer---\n";
			runNormalizer();
			break;
		case (QUIT_OPTION):
		default:
			keepRunning = false;
			break;
		}
	}

	return true;
}

bool ArcCli::runMatrixMultiplication()
{
	ArcMatrixMultiply matrixMultiply;

	if (!matrixMultiply.performMatrixMultiplication())
	{
		return false;
	}

	return true;
}

bool ArcCli::runNormalizer()
{
	ArcNormalization normalizer;

	for (int i = 0; i < 10; ++i)
	{
		std::cout << "\n----------Normalizer Iteration " << i << "----------\n";

		if (!normalizer.performNormalization())
		{
			return false;
		}
	}

	return true;
}
