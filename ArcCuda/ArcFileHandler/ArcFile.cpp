#include <fstream>
#include <iostream>

#include "ArcFile.h"

// Constructor(s) //

ArcFile::ArcFile()
{
}

ArcFile::ArcFile(std::string& fileName)
{
}

// Destructor(s)

ArcFile::~ArcFile()
{
}

// Methods - Public //



// Methods - Private //

std::streampos ArcFile::getFileSize()
{
	std::streampos fsize = 0;
	std::ifstream file(_fileName, std::ios::binary);

	fsize = file.tellg();
	file.seekg(0, std::ios::end);
	fsize = file.tellg() - fsize;
	file.close();

	return fsize;
}
