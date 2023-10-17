#ifndef ARC_FILE_H
#define ARC_FILE_H

#include <vector>
#include <string>

class ArcFile
{
public: // Constructor(s) //

	ArcFile();

	ArcFile(std::string& fileName);

public: // Destructor(s) //

	~ArcFile();

public: // Methods - Public //

	bool readFile();

private: // Methods - Private //

	std::streampos getFileSize();

private: // Variables - Private //

	std::string _fileName;
};

#endif // !ARC_FILE_H
