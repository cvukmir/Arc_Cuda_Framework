#ifndef ARC_NORMALIZATION_H
#define ARC_NORMALIZATION_H


class ArcNormalization
{
public: // Constructor(s) //

	ArcNormalization();

public: // Destructor(s) //

	~ArcNormalization();

public: // Methods - Public //

	bool performNormalization();

private: // Methods - Private //

	void clearArray();

	void fillArray();

	void initializeArray();

	void initializeSize();

	bool normalizeCpu();

	bool normalizeGpu();

	void printArray();

private: // Variables - Private //

	float* _pArray;
	float  _normalizedValue;
	int    _size;
};


#endif // !ARC_NORMALIZATION_H