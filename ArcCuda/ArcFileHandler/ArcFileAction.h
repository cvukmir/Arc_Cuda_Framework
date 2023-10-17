#ifndef ARC_FILE_ACTION_H
#define ARC_FILE_ACTION_H

class ArcFileAction
{
public:

	// Constructors //

	ArcFileAction();

	// Destructors //

	~ArcFileAction();

	// Public Methods - Static //

	static void readCsv();

	// Private Methods //

	bool readFile(const std::string& fileName);

};
#endif // !ARC_FILE_ACTION_H