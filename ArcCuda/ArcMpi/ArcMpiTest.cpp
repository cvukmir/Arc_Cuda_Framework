#include "mpi.h"

void test(int* argc, char*** argv)
{
	int test = MPI_Init(argc, argv);

	MPI_Finalize();

}