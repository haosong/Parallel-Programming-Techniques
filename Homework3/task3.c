#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "timing.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define BLOCK_LEN(rowIndex, blockSize) ((2*rowIndex+blockSize+1)*blockSize/2)

double block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int blockSize, int N);
void swap(double** A, double** B);

int main(int argc, char **argv) {
    int sizes[4]={1000,2000,4000,8000};
    char files[4][50]={"/home/fas/cpsc424/ahs3/assignment3/C-1000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-2000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-4000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-8000.dat"};
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0)
        printf("Matrix multiplication times:\n RANK   COMP-TIME (secs)   COMM-TIME (secs)   TIME (secs)\n -----   -----------------   -----------------   -------------\n");
    
    int run = (argc == 2) ? 3 : 0; // if have argument, just run for N = 8000; otherwise, run all.
    for (; run < 4; run++) {
        double *A, *B, *C, *Ctrue;
        double wcs, wce, ct;
        FILE *fptr;
        int N = sizes[run];
        int blockSize = N / size;
        int sizeAB = N * (N + 1) / 2;
        int sizeC = N * N;
        double compTime = 0, commTime = 0;

        if (rank == 0) {
            printf("N = %d\n", N);
            // Declare MPI status and request
            MPI_Status status;
            MPI_Request sendRequest[2 * size], recvRequest[size];
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double));
            C = (double *) calloc(sizeC, sizeof(double));
            // Generate A and B
            srand(12345);
            for (int i = 0; i < sizeAB; i++) A[i] = ((double) rand() / (double) RAND_MAX);
            for (int i = 0; i < sizeAB; i++) B[i] = ((double) rand() / (double) RAND_MAX);
            MPI_Barrier(MPI_COMM_WORLD);

            // Send the permanent row alignment to all workers
            timing(&wcs, &ct);
            for (int i = 1; i < size; i++) {
                int rowIndex = i * blockSize;
                int length = BLOCK_LEN(rowIndex, blockSize);
                MPI_Isend(A + rowIndex * (rowIndex + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &sendRequest[i]);
            }
            for(int i = 1; i < size; i++)
                MPI_Wait(&sendRequest[i], &status);            
            // Send column to all workers
            for (int i = 1; i < size; i++) {
                int colIndex = i * blockSize;
                int length = BLOCK_LEN(colIndex, blockSize);
                MPI_Isend(&colIndex, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &sendRequest[i]);
                MPI_Isend(B + colIndex * (colIndex + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &sendRequest[i + size]);
            }
            for (int i = 1; i < size; i++) {
                MPI_Wait(&sendRequest[i], &status);
                MPI_Wait(&sendRequest[i + size], &status);
            }
            // The above part have to wait, which make it equivallent to blocking implementation.
            // But doesn't affect the general optimization below.

            int rowIndex = 0;
            int colIndex = 0;
            int nextColIndex;
            double *col = B;
            double *nextB = (double *) calloc(sizeAB, sizeof(double)); // buffer for next col
            double *freeNextB = nextB;
            // Send column to next node, receive column from prev node
            for (int i = 1; i < size; i++) {
                MPI_Isend(&colIndex, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &sendRequest[0]);
                MPI_Isend(col, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &sendRequest[1]);
                MPI_Irecv(&nextColIndex, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &recvRequest[0]);
                MPI_Irecv(nextB, sizeAB, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD, &recvRequest[1]);
                // Do the calculation while recving next block, key part of non-blocking implementation.
                compTime += block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);
                MPI_Wait(&sendRequest[0], &status);
                MPI_Wait(&sendRequest[1], &status);
                MPI_Wait(&recvRequest[0], &status);
                MPI_Wait(&recvRequest[1], &status);
                // Shouldn't wait too long above, since we're receving data while doing the calculation
                colIndex = nextColIndex;
                swap(&col, &nextB);
            }

            // Collect results from workers
            for (int i = 1; i < size; i++)
                MPI_Irecv(C + i * blockSize * N, blockSize * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &recvRequest[i]);
            // Doing the last block calculation while collect results
            compTime += block_matmul(A, col, C, rowIndex, colIndex, blockSize, N);
            for (int i = 1; i < size; i++)
                MPI_Wait(&recvRequest[i], &status);
            
            timing(&wce, &ct);
            commTime = wce - wcs - compTime;
            free(A);
            free(B);
            // Read correct answer
            Ctrue = (double *) calloc(sizeC, sizeof(double));
            fptr = fopen(files[run], "rb");
            fread(Ctrue, sizeof(double), sizeC, fptr);
            fclose(fptr);
            // Checking
            double Fnorm = 0.;
            for (int i = 0; i < N * N; i++) Fnorm += (Ctrue[i] - C[i]) * (Ctrue[i] - C[i]);
            Fnorm = sqrt(Fnorm);
            printf("  %5d    %9.4f    %9.4f    %9.4f\n", 0, compTime, commTime, wce - wcs);
            printf("F-norm of Error: %15.10f\n", Fnorm);
            printf("Total Runtime: %9.4f\n", wce - wcs);
            free(Ctrue);
            free(C);
            free(freeNextB);
        } else {
            MPI_Status status;
            MPI_Request sendRequest[size], recvRequest[size];
            int colIndex, nextColIndex;
            int rowIndex = blockSize * rank;
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double));
            C = (double *) calloc(sizeC, sizeof(double));
            double *nextB = (double *) calloc(sizeAB, sizeof(double)); // buffer for next col
            MPI_Barrier(MPI_COMM_WORLD);
            timing(&wcs, &ct);
            MPI_Irecv(A, BLOCK_LEN(rowIndex, blockSize), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &recvRequest[0]);
            MPI_Irecv(&colIndex, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recvRequest[1]);
            MPI_Irecv(B, sizeAB, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &recvRequest[2]);
            MPI_Wait(&recvRequest[0], &status);
            MPI_Wait(&recvRequest[1], &status);
            MPI_Wait(&recvRequest[2], &status);
            // Receive row data, have to wait.
            for (int i = 1; i < size; i++) {
                MPI_Isend(&colIndex, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD, &sendRequest[0]);
                MPI_Isend(B, BLOCK_LEN(colIndex, blockSize), MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD, &sendRequest[1]);
                MPI_Irecv(&nextColIndex, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &recvRequest[0]);
                MPI_Irecv(nextB, sizeAB, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &recvRequest[1]);
                // Do the calculation while recving next block, key part of non-blocking implementation.
                compTime += block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);                
                MPI_Wait(&recvRequest[0], &status);
                MPI_Wait(&recvRequest[1], &status);
                MPI_Wait(&sendRequest[0], &status);
                MPI_Wait(&sendRequest[1], &status);
                colIndex = nextColIndex;
                //B = nextB;
                swap(&B, &nextB);
            }
            compTime += block_matmul(A, B, C, rowIndex, colIndex, blockSize, N);
            MPI_Send(C, N * blockSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

            timing(&wce, &ct);
            commTime = wce - wcs - compTime;
            printf("  %5d    %9.4f    %9.4f    %9.4f\n", rank, compTime, commTime, wce - wcs);

            free(A);
            free(B);
            free(C);
            free(nextB);
        }
    }

    MPI_Finalize();

}

// Do matrix multiplication for certain block
double block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int blockSize, int N) {
    double wctime0, wctime1, cputime;
    timing(&wctime0, &cputime);
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
    timing(&wctime1, &cputime);
    return(wctime1 - wctime0);
}

void swap(double** A, double** B) {
    double *tmp = *A;
    *A = *B;
    *B = tmp;
}
