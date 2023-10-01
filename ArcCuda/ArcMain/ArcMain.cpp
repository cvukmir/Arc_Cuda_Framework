#include "ArcAssignment1.h"
#include "ArcAssignment2.h"


// C:\Users\Chris\source\repos\Cuda_Test_1
//#include "C:\Users\Chris\source\repos\Cuda_Test_1\"

int main()
{
	if (!ArcAssignment1::runAssignment1())
	{
		return -1;
	}

	ArcAssignment2 Assign2;

	if (!Assign2.runAssignment2())
	{
		return -1;
	}

	return 0;
}