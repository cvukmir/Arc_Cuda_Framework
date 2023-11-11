// Mpi
#include "mpi.h"

// ArcMpi
#include "ArcMpi.h"

#include <cstring>
#include <iostream>
#include <iomanip>
#include <cstdlib>

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

// Part3:
// Broadcast where rank 0 sends a value to every thread but efficiently
// Rank 0 sends to 1 and both send to two more which send to 8 more and so on.
//

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	
	int rankCount = 0;
	int myRank    = 0;

	int value = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

	if (myRank == 0)
	{
		std::cout << "Rank Count: " << rankCount << std::endl;

		std::cout << std::endl << "---Part 1---" << std::endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD);

	ArcMpi::part1(myRank, rankCount);

	//MPI_Barrier(MPI_COMM_WORLD);

	if (myRank == 0)
	{
		std::cout << std::endl << "---Part 2---" << std::endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD);

	ArcMpi::part2(myRank, rankCount);

	//MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

	return 0;
}

void ArcMpi::part1(const int myRank, const int rankCount)
{
	if (rankCount == 1)
	{
		std::cout << "Only one rank called, part 1 not executed." << std::endl;
		return;
	}

	MPI_Status status;
	int value = 0;

	if (myRank == 0)
	{
		value = 0;

		std::cout << "Rank " << myRank << " initial value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank    + 1, 99, MPI_COMM_WORLD);
		MPI_Recv(&value, 1, MPI_INT, rankCount - 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved value: " << value << std::endl;
	}
	else if (myRank == rankCount - 1)
	{
		MPI_Recv(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved value: " << value << std::endl;

		value += 1;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, 0,    99, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved a value: " << value << std::endl;

		value += 1;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank + 1, 99, MPI_COMM_WORLD);
	}
}

void ArcMpi::part2(const int myRank, const int rankCount)
{
	if (rankCount == 1)
	{
		std::cout << "Only one rank called, part 2 not executed." << std::endl;
		return;
	}

	MPI_Status status;
	int value = 0;

	if (myRank == 0)
	{
		value = 0;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank    + 1, 99, MPI_COMM_WORLD);
		MPI_Recv(&value, 1, MPI_INT, myRank + 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved value: " << value << std::endl;
	}
	else if (myRank == rankCount - 1)
	{
		MPI_Recv(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved value: " << value << std::endl;

		value -= 1;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved a value: " << value << std::endl;

		value += 1;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank + 1, 99, MPI_COMM_WORLD);

		MPI_Recv(&value, 1, MPI_INT, myRank + 1, 99, MPI_COMM_WORLD, &status);

		std::cout << "Rank " << myRank << " recieved value: " << value << std::endl;

		value -= 1;

		std::cout << "Rank " << myRank << " sending value: " << value << std::endl;

		MPI_Send(&value, 1, MPI_INT, myRank - 1, 99, MPI_COMM_WORLD);
	}
}
