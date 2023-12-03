#ifndef ARC_MPI_H
#define ARC_MPI_H

class ArcMpi
{
public:
	static void assignment1();
	static void assignment2();
	static void assignment3();

	static void baseline(const int myRank, const int rankCount, int* value, const int valueCount);

	static void broadcastSend(const int myRank, const int rankCount, const int* value, const int valueCount);
	static void broadcastRecv(const int myRank, const int rankCount, int* value, const int valueCount);

	static void part1(const int myRank, const int rankCount);
	static void part2(const int myRank, const int rankCount);
	
	static void sequentialSend(const int myRank, const int rankCount, const int* value, const int valueCount);
	static void sequentialRecv(const int myRank, const int rankCount, int* value, const int valueCount);
};

#endif // !ARC_MPI_H
