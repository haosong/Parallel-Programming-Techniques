#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "timing.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define BLOCK_LEN(rowIndex, blockSize) ((2*rowIndex+blockSize+2)*blockSize/2)

void block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int blockSize, int N);

int main(int argc, char **argv) {
    int rank, size;
    double *A, *B, *C;
    // int sizeAB, sizeC;
    int N = atoi(argv[1]);

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int blockSize = N / size;

    // sizeAB = N * (N + 1) / 2;
    // sizeC = N * N;

    if (rank == 0) {
        A = (double *) calloc(N * (N + 1) / 2, sizeof(double));
        B = (double *) calloc(N * (N + 1) / 2, sizeof(double));
        C = (double *) calloc(N * N, sizeof(double));

        srand(12345);

        for (int i = 0; i < N * (N + 1) / 2; i++) A[i] = ((double) rand() / (double) RAND_MAX);
        for (int i = 0; i < N * (N + 1) / 2; i++) B[i] = ((double) rand() / (double) RAND_MAX);

        MPI_Barrier(MPI_COMM_WORLD);

        // Send the permanent row to all workers
        for (int i = 1; i < size; i++) {
            int rowIndex = i * blockSize;
            int length = BLOCK_LEN(rowIndex, blockSize);;
            MPI_Send(A + rowIndex * (rowIndex + 1) / 2, length, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD);
        }

        // Send column to all workers
        for (int i = 1; i < size; i++) {
            int colIndex = i * blockSize;
            int length = BLOCK_LEN(colIndex, blockSize);;
            MPI_Send(&colIndex, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD);
            MPI_Send(B + colIndex * (colIndex + 1) / 2, length, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD);
        }

        // Do the calculation on currant master node
        int rowIndex = 0;
        int colIndex = 0;
        double *col = B;
        block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);

        // Send column to next node, receive column from prev node
        for (int i = 1; i < size; i++) {
            MPI_Send(&colIndex, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
            MPI_Send(col, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD);
            MPI_Recv(&colIndex, 1, MPI_INT, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(col, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // Calculate on those coming columns
            block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);
        }

        // Collect results from workers
        for (int i = 1; i < size; i++)
            MPI_Recv(C + i * blockSize * N, blockSize * N, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        free(A);
        free(B);
        free(C);
    } else {
        int colIndex;
        int rowIndex = blockSize * rank;
        A = (double *) calloc(BLOCK_LEN(rowIndex, blockSize), sizeof(double));
        B = (double *) calloc((2 * N - blockSize + 1) * blockSize / 2,
                              sizeof(double)); // Most elements containing in the last block
        C = (double *) calloc(N * blockSize, sizeof(double));
        double *nextB = (double *) calloc(N * blockSize, sizeof(double));

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Recv(A, BLOCK_LEN(rowIndex, blockSize), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&colIndex, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(B, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);

        for (int i = 1; i < size; i++) {
            int nextColIndex;
            MPI_Recv(&nextColIndex, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(nextB, BLOCK_LEN(nextColIndex, blockSize), MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD,
                     &status);
            MPI_Send(&colIndex, 1, MPI_INT, (rank + 1) % size, MPI_ANY_TAG, MPI_COMM_WORLD);
            MPI_Send(B, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, (rank + 1) % size, MPI_ANY_TAG, MPI_COMM_WORLD);
            colIndex = nextColIndex;
            memcpy(B, nextB, sizeof(double) * BLOCK_LEN(colIndex, blockSize));
            block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);
        }

        MPI_Send(C, N * blockSize, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD);

        free(A);
        free(B);
        free(C);
        free(nextB);
    }

    MPI_Finalize();
}

void block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int blockSize, int N) {
    int iA, iB, iC;
    for (int i = 0; i < blockSize; i++) {
        iC = i * N + colIndex;
        iA = (2 * rowIndex + i + 1) * i / 2;
        for (int j = 0; j < blockSize; j++, iC++) {
            iB = (2 * colIndex + j + 1) * j / 2;
            C[iC] = 0;
            for (int k = 0; k <= MIN(i + rowIndex, j + colIndex); k++) C[iC] += A[iA + k] * B[iB + k];
        }
    }
}
