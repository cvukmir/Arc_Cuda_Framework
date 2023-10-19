#include "ArcMatrix.h"
#include <stdint.h>


// Constructor(s) //

template <typename T>
ArcMatrix<T>::ArcMatrix(int sizeX, int sizeY)
	: _x       (sizeX)
	, _y       (sizeY)
{
	_pMatrix = new T[_x * _y];
}

// Destructor(s) //

template <typename T>
ArcMatrix<T>::~ArcMatrix()
{
	if (_pMatrix)
	{
		delete _pMatrix;
	}
}

// Properties - Public //

template<typename T>
int ArcMatrix<T>::columnCount() const { return _x; }

template<typename T>
int ArcMatrix<T>::rowCount() const { return _y; }

template<typename T>
int ArcMatrix<T>::x() const { return _x; }

template<typename T>
int ArcMatrix<T>::y() const { return _y; }

// Methods - Public //

template<typename T>
bool ArcMatrix<T>::matrixMultiplyCpu()
{
	return true;
}

template<typename T>
bool ArcMatrix<T>::matrixMultiplyGpu()
{
	return true;
}

