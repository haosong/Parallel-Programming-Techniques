#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "timing.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define BLOCK_LEN(rowIndex, blockSize) ((2*rowIndex+blockSize+2)*blockSize/2)

void block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int blockSize, int N);

int main(int argc, char **argv) {
    int sizes[4]={1000,2000,4000,8000};

    char files[4][50]={"/home/fas/cpsc424/ahs3/assignment3/C-1000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-2000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-4000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-8000.dat"};
    int rank, size;
    double *A, *B, *C, *Ctrue;
    int N, sizeAB, sizeC;
    FILE *fptr;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int run = 0; run < 4; run++) {
        N = sizes[run];

        int blockSize = N / size;
        sizeAB = N * (N + 1) / 2;
        sizeC = N * N;

        if (rank == 0) {
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double));
            C = (double *) calloc(sizeC, sizeof(double));
            srand(12345);

            for (int i = 0; i < sizeAB; i++) A[i] = ((double) rand() / (double) RAND_MAX);
            for (int i = 0; i < sizeAB; i++) B[i] = ((double) rand() / (double) RAND_MAX);
            MPI_Barrier(MPI_COMM_WORLD);

            // Send the permanent row to all workers
            for (int i = 1; i < size; i++) {
                int rowIndex = i * blockSize;
                int length = BLOCK_LEN(rowIndex, blockSize);
                MPI_Send(A + rowIndex * (rowIndex + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }

            // Send column to all workers
            for (int i = 1; i < size; i++) {
                int colIndex = i * blockSize;
                int length = BLOCK_LEN(colIndex, blockSize);;
                MPI_Send(&colIndex, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(B + colIndex * (colIndex + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }

            // Do the calculation on currant master node
            int rowIndex = 0;
            int colIndex = 0;
            double *col = B;
            block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);

            // Send column to next node, receive column from prev node
            for (int i = 1; i < size; i++) {
                MPI_Send(&colIndex, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
                MPI_Send(col, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
                MPI_Recv(&colIndex, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(col, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD, &status);
                // Calculate on those coming columns
                block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);
            }

            // Collect results from workers
            for (int i = 1; i < size; i++)
                MPI_Recv(C + i * blockSize * N, blockSize * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);

            Ctrue = (double *) calloc(sizeC, sizeof(double));
            fptr = fopen(files[run], "rb");
            fread(Ctrue, sizeof(double), sizeC, fptr);
            fclose(fptr);

            double Fnorm = 0.;
            for (int i = 0; i < N * N; i++) Fnorm += (Ctrue[i] - C[i]) * (Ctrue[i] - C[i]);
            Fnorm = sqrt(Fnorm);
            printf("  %5d    %9.4f  %15.10f\n", N, 0.12345, Fnorm);

            free(A);
            free(B);
            free(C);
        } else {
            int colIndex;
            int rowIndex = blockSize * rank;
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double)); // Most elements containing in the last block
            C = (double *) calloc(sizeC, sizeof(double));
            double *nextB;
            nextB = (double *) calloc(sizeAB, sizeof(double));

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Recv(A, BLOCK_LEN(rowIndex, blockSize), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&colIndex, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(B, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);

            for (int i = 1; i < size; i++) {
                int nextColIndex;
                MPI_Recv(&nextColIndex, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(nextB, BLOCK_LEN(nextColIndex, blockSize), MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
                MPI_Send(&colIndex, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD);
                MPI_Send(B, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD);
                colIndex = nextColIndex;
                memcpy(B, nextB, sizeof(double) * BLOCK_LEN(colIndex, blockSize));
                block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);
            }

            MPI_Send(C, N * blockSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

            free(A);
            free(B);
            free(C);
            free(nextB);
        }
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
