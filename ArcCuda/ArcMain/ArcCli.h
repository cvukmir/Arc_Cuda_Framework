#ifndef ARC_CLI_H
#define ARC_CLI_H

class ArcCli
{
public: // Static Methods - Public //

	static bool runInterface();

private: // Static Methods - Private //
	
	static bool runMatrixMultiplication();

	static bool runNormalizer();

private: // Static Variables - Private //

	static const char INVALID_OPTION         = 'i';
	static const char MATRIX_MULTIPLY_OPTION = 'm';
	static const char NORMALIZER_OPTION      = 'n';
	static const char QUIT_OPTION            = 'q';
};

#endif // !ARC_CLI_H