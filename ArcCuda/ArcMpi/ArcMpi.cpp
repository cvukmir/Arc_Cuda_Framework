// Mpi
#include "mpi.h"

// ArcMpi
#include "ArcMpi.h"

#include <cstring>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

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

// Part 4:
// Conways game of life
// Large grid
// Rank 0 is in charge of gathering pieces and generating result
// Netpbm
// The magic = metadata to say what file type aka sigatures
// IrfanView - image file viewer
// 
// 

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	
	//ArcMpi::assignment1();

	//ArcMpi::assignment2();

	ArcMpi::assignment3();

	MPI_Finalize();

	return 0;
}

void ArcMpi::assignment1()
{
	int rankCount = 0;
	int myRank    = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

	if (myRank == 0)
	{
		std::cout << "Rank Count: " << rankCount << std::endl;

		std::cout << std::endl << "---Part 1---" << std::endl;
	}

	ArcMpi::part1(myRank, rankCount);

	if (myRank == 0)
	{
		std::cout << std::endl << "---Part 2---" << std::endl;
	}

	ArcMpi::part2(myRank, rankCount);
}

void ArcMpi::assignment2()
{
	int rankCount = 0;
	int myRank    = 0;
	const int size = 10000;
	int sendValue[size];
	int recvValue[size];

	for (int i = 0; i < size; ++i)
	{
		sendValue[i] = 100;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

	//////////////// TEST 1 ///////////////

	//MPI_Barrier(MPI_COMM_WORLD);
	//
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//
	//if (myRank == 0)
	//{
	//	baseline(myRank, rankCount, sendValue, size);
	//}
	//else
	//{
	//	baseline(myRank, rankCount, recvValue, size);
	//}
	
	//MPI_Barrier(MPI_COMM_WORLD);
	//
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//
	//if (myRank == 0)
	//{
	//	std::cout << "Baseline broadcast send/recv in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds." << '\n';
	//}

	//////////////// TEST 2 ///////////////

	MPI_Barrier(MPI_COMM_WORLD);
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	if (myRank == 0)
	{
		broadcastSend(myRank, rankCount, sendValue, size);
	}
	else
	{
		broadcastRecv(myRank, rankCount, recvValue, size);
	}
	
	//MPI_Barrier(MPI_COMM_WORLD);
	
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	
	//if (myRank == 0)
	//{
		std::cout << "Sent data to ranks exponentially in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds." << '\n';
	//}

	//////////////// TEST 3 ///////////////

	//MPI_Barrier(MPI_COMM_WORLD);
	//
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//
	//if (myRank == 0)
	//{
	//	sequentialSend(myRank, rankCount, sendValue, size);
	//}
	//else
	//{
	//	sequentialRecv(myRank, rankCount, recvValue, size);
	//}
	//
	//MPI_Barrier(MPI_COMM_WORLD);
	//
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//
	//if (myRank == 0)
	//{
	//	std::cout << "Sent data to ranks sequentially in " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds." << '\n';
	//}
}

void ArcMpi::assignment3()
{
	// Split up tasks based on ranks.
	int rankCount      = 0;
	int myRank         = 0;
	int gameIterations = 10;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

	if (myRank == 0)
	{
		// Initialize the grid
		size_t arraySize = 10;
		bool* pMatrix = new bool[arraySize * arraySize];

		srand(unsigned int(time(NULL)));

		for (int rowIndex = 0; rowIndex < arraySize; ++rowIndex)
		{
			for (int columnIndex = 0; columnIndex < arraySize; ++columnIndex)
			{
				pMatrix[rowIndex * arraySize + columnIndex] = rand() % 2 == 0 ? false : true;
			}
		}

		std::cout << "Printing the initial grid state." << '\n';

		for (int rowIndex = 0; rowIndex < arraySize; ++rowIndex)
		{
			for (int columnIndex = 0; columnIndex < arraySize; ++columnIndex)
			{
				std::cout << pMatrix[rowIndex * arraySize + columnIndex] << ' ';
			}

			std::cout << '\n';
		}

		for (int iterationNumber = 1; iterationNumber <= gameIterations; ++iterationNumber)
		{
			std::cout << "Starting game iteration " << iterationNumber << '\n';

			// Figure out optimal way to split up data.


			std::cout << "Print game state after iteration " << iterationNumber << '\n';

			for (int rowIndex = 0; rowIndex < arraySize; ++rowIndex)
			{
				for (int columnIndex = 0; columnIndex < arraySize; ++columnIndex)
				{
					std::cout << pMatrix[rowIndex * arraySize + columnIndex] << ' ';
				}

				std::cout << '\n';
			}
		}

		std::cout << "Closing game." << '\n';

		delete[] pMatrix;
	}
	else
	{

	}
}

void ArcMpi::baseline(const int myRank, const int rankCount, int* value, const int size)
{
	MPI_Bcast(value, size, MPI_INT, 0, MPI_COMM_WORLD);
}

void ArcMpi::broadcastSend(const int myRank, const int rankCount, const int* value, const int size)
{
	if (rankCount < 2)
	{
		return;
	}

	// This rank is always even.
	for (int i = 1; i < rankCount; i = (i << 1) + 1)
	{
		MPI_Send(value, size, MPI_INT, i, 99, MPI_COMM_WORLD);
		std::cout << "Rank " << myRank << " sent value " << value << " to rank " << i << std::endl;
	}
}

void ArcMpi::broadcastRecv(const int myRank, const int rankCount, int* value, const int size)
{
	MPI_Status status;

	//std::cout << "Rank " << myRank << " wating..." << std::endl;
	MPI_Recv(value, size, MPI_INT, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &status);
	//std::cout << "Rank " << myRank << " recieved value " << value << std::endl;

	if (myRank & 0x00000001)
	{
		// This rank is odd.
		for (int i = myRank << 1; i < rankCount; i = (i << 1))
		{
			MPI_Send(value, size, MPI_INT, i, 99, MPI_COMM_WORLD);
			std::cout << "Rank " << myRank << " sent value " << value << " to rank " << i << std::endl;
		}
	}
	else
	{
		// This rank is even.
		for (int i = (myRank << 1) + 1; i < rankCount; i = (i << 1) + 1)
		{
			MPI_Send(value, size, MPI_INT, i, 99, MPI_COMM_WORLD);
			std::cout << "Rank " << myRank << " sent value " << value << " to rank " << i << std::endl;
		}
	}
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

void ArcMpi::sequentialSend(const int myRank, const int rankCount, const int* value, const int size)
{
	for (int i = 1; i < rankCount; ++i)
	{
		MPI_Send(value, size, MPI_INT, i, 99, MPI_COMM_WORLD);
		//std::cout << "Rank " << myRank << " sent value " << value << " to rank " << i << std::endl;
	}
}

void ArcMpi::sequentialRecv(const int myRank, const int rankCount, int* value, const int size)
{
	MPI_Status status;
	MPI_Recv(value, size, MPI_INT, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &status);
}
