#ifndef ARC_MATRIX_MULTIPLY_H
#define ARC_MATRIX_MULTIPLY_H

class ArcMatrixMultiply
{
public: // Constructor(s) //

	ArcMatrixMultiply();
	
public: // Destructor(s) //

	~ArcMatrixMultiply();

public: // Methods - Public //

	// Executes the methods need for assignment 2.
	bool performMatrixMultiplication();

private: // Methods - Private //

	// Zeros the given matrix that has the given sizes.
	void clearMatrix(float* pMatrix, const int rowSize, const int columnSize);

	// Calculates the dot product from the two matricies at the row index of the first matrix and column index of the second for the given size.
	float dotProduct(float* pMatrix1, float* pMatrix, const int rowIndex, const int columnIndex, const int size);

	// Fills the matrix with random values of the specified number of rows and columns.
	void fillMatrix(float* pMatrix, const int numberOfRows, const int numberOfColumns);

	// Generates this objects three matricies.
	void generateMatrices();

	// Initializes the class matricies to the sizes needed for matrix multiplication.
	void initializeMatrices();

	// Initializes the class matricies sizes to random sizes.
	void initializeSizes();

	// Preforms matrix multiplication on this objects three matricies using the CPU.
	void multiplyMatricesCPU();

	// Preforms matrix multiplication on this objects three matricies using the GPU.
	bool multiplyMatricesGPU();

	// Prints the given matrix with its specified number of rows and columns.
	void printMatrix(float* pMatrix, const int numberOfRows, const int numberOfColumns);

private: // Variables - Private //

	float* _pMatrix1;    // This objects first  matrix.
	float* _pMatrix2;    // This objects second matrix.
	float* _pMatrix3;    // This objects third  matrix.
	int    _matrixSizeM; // This objects row    size of the first  matrix and row    size of the third  matrix.
	int    _matrixSizeN; // This objects column size of the first  matrix and row    size of the second matrix.
	int    _matrixSizeP; // This objects column size of the second matrix and column size of the third  matrix.
};

#endif // !ARC_MATRIX_MULTIPLY_H