// ArcMain
#include "ArcMatrixMultiply.h"
#include "ArcNormalization.h"

// ArcMpi
#include "ArcMpi.h"

// Std C++
#include <iostream>


int main(int argc, char* argv[])
{
	
//	ArcMatrixMultiply matrixMultiply;

//	if (!matrixMultiply.performMatrixMultiplication())
//	{
//		return -1;
//	}

//	ArcNormalization normalizer;
	
//	for (int i = 0; i < 10; ++i)
//	{
//		std::cout << "\n----------Normalizer Iteration " << i << "----------\n";
//
//		if (!normalizer.performNormalization())
//		{
//			return -1;
//		}
//	}

	//int test = 0;
	//char* test2[1];
	ArcMpi test;
	test.test(&argc, &argv);

	return 0;
}