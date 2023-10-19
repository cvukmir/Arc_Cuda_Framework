// Std C++
#include <string>
#include <fstream>
#include "ArcFileAction.h"

#include "ArcFile.h"


// Constructor(s) //

ArcFileAction::ArcFileAction()
{
}

// Destructor(s) //

ArcFileAction::~ArcFileAction()
{
}

// Static Methods - Public //

void ArcFileAction::readCsv()
{
	
}

// Methods - Private //

bool ArcFileAction::readFile(const std::string& fileName)
{
	std::fstream fileStream = std::fstream(fileName);

	if (!fileStream.is_open())
	{
		return false;
	}

	//fileStream.read();

	fileStream.flush();
	fileStream.close();

	return true;
}
