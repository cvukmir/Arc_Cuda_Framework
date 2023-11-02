// ArcMain
#include "ArcMatrixMultiply.h"
#include "ArcNormalization.h"


#include <wmpids.h>
// Std C++
#include <iostream>


int main()
{
	
//	ArcMatrixMultiply matrixMultiply;

//	if (!matrixMultiply.performMatrixMultiplication())
//	{
//		return -1;
//	}

	ArcNormalization normalizer;
	
	for (int i = 0; i < 10; ++i)
	{
		std::cout << "\n----------Normalizer Iteration " << i << "----------\n";

		if (!normalizer.performNormalization())
		{
			return -1;
		}
	}

	return 0;
}