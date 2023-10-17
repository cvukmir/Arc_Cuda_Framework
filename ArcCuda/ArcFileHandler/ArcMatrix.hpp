#ifndef ARC_MATRIX_H
#define ARC_MATRIX_H

#include <stdint.h>

template <typename T>
struct ArcMatrix
{
public: // Constructor(s) //

	ArcMatrix(int64_t sizeX, int64_t sizeY)
	{
		columnSize = sizeX;
		rowSize    = sizeY;

		pMatrix = new T[rowSize * columnSize];
	}

public: // Destructor(s) //

	~ArcMatrix()
	{
		delete pMatrix;
	}

public: // Varibles - Public //

	T*      pMatrix;
	int64_t columnSize;
	int64_t rowSize;
};

#endif // !ARC_MATRIX_H