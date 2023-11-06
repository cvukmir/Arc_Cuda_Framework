#ifndef ARC_MPI_H
#define ARC_MPI_H

class ArcMpi
{
public:
	static void test(int* argc, char*** argv);

	static void part1(const int myRank, const int rankCount);
	static void part2(const int myRank, const int rankCount);
};

#endif // !ARC_MPI_H
