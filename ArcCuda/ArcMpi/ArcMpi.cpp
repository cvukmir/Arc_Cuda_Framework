// Mpi
#include "mpi.h"

// ArcMpi
#include "ArcMpi.h"

#include <cstring>
#include <iostream>

// Part 1:
// Rank 0 sends an int 0 to rank 1 and prints statement
// Rank 1 recieves from 0, increments the value and prints statement.
// Rank 1 sends same data to rank 2 and prints statement.
// Loop until end 
// Rank n sends to 0.

// Part 2:
// Do same until rank n.
// instead of sending to rank 0, send back down the rank numbers.
// Decrement the sent value.

int main(int argc, char* argv[])
{
	ArcMpi::test(&argc, &argv);
	return 0;
}

void ArcMpi::test(int* argc, char*** argv)
{
	char message[20];
	int myrank;
	MPI_Status status;
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	std::cout << "Hello my rank is: " << myrank << "\n";
	//MPI_Comm_size(MPI_COMM_WORLD, )
	//std::cout << "Comm Size: " <<
	/*
	if (myrank == 0) 
	{
		strcpy_s(message, "Hello, there");
		MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 99, MPI_COMM_WORLD);
	}
	else if (myrank == 1)
	{
		MPI_Recv(message, 20, MPI_CHAR, 0, 99, MPI_COMM_WORLD, &status);
		printf("received :%s:\n", message);
	}*/
	
	MPI_Finalize();
}