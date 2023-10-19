// ArcMain
#include "ArcMatrixMultiply.h"
#include "ArcNormalization.h"


int main()
{
//	ArcMatrixMultiply matrixMultiply;

//	if (!matrixMultiply.performMatrixMultiplication())
//	{
//		return -1;
//	}

	ArcNormalization normalizer;

	if (!normalizer.performNormalization())
	{
		return -1;
	}

	return 0;
}