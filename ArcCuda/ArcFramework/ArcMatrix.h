#ifndef ARC_MATRIX_H
#define ARC_MATRIX_H

#include <stdint.h>

template <typename T>
class ArcMatrix
{
public: // Constructor(s) //

	ArcMatrix(int sizeX, int sizeY);

public: // Destructor(s) //

	~ArcMatrix();

public: // Properties - Public //

	int  columnCount() const; // Gets the column count (x) of the matrix.

	int  rowCount() const;    // Gets the row    count (y) of the matrix.

	int  x() const;           // Gets the number of columns in the matrix.

	int  y() const;           // Gets the number of    rows in the matrix.
	
	


public: // Methods - Public //

	bool matrixMultiplyCpu();

	bool matrixMultiplyGpu();


private: // Varibles - Private //

	T*  _pMatrix;
	int _x;
	int _y;
};

#endif // !ARC_MATRIX_H