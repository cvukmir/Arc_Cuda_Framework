#ifndef ARC_ASSIGNMENT_2_H
#define ARC_ASSIGNMENT_2_H

class ArcAssignment2
{
public:

	// Constructors //

	ArcAssignment2();
	~ArcAssignment2();

	// Public Methods //

	// Executes the methods need for assignment 2.
	bool runAssignment2();

private:

	// Private Methods //

	// Calculates the dot product from the two matricies at the row index of the first matrix and column index of the second for the given size.
	float dotProduct(float* ppMatrix1, float* ppMatrix, const int rowIndex, const int columnIndex, const int size);

	// Fills the matrix with random values of the specified number of rows and columns.
	void fillMatrix(float* ppMatrix, const int numberOfRows, const int numberOfColumns);

	// Generates this objects three matricies.
	void generateMatrices();

	void initializeMatrices();

	void initializeSizes();

	// Preforms matrix multiplication on this objects three matricies using the CPU.
	void multiplyMatricesCPU();

	// Preforms matrix multiplication on this objects three matricies using the GPU.
	bool multiplyMatricesGPU();

	// Prints the given matrix with its specified number of rows and columns.
	void printMatrix(float* ppMatrix, const int numberOfRows, const int numberOfColumns);

	// Private Variables //

	float* _pMatrix1;    // This objects first  matrix.
	float* _pMatrix2;    // This objects second matrix.
	float* _pMatrix3;    // This objects third  matrix.
	int    _matrixSizeM; // This objects row    size of the first  matrix and row    size of the third  matrix.
	int    _matrixSizeN; // This objects column size of the first  matrix and row    size of the second matrix.
	int    _matrixSizeP; // This objects column size of the second matrix and column size of the third  matrix.
};

#endif // !ARC_ASSIGNMENT_2_H