#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "timing.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define BLOCK_LEN(rowIndex, blockSize) ((2*rowIndex+blockSize+1)*blockSize/2)

void block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int rowBlockSize, int colBlockSize, int N);
void swap(double** A, double** B);

int main(int argc, char **argv) {
    int sizes[5]={1000,2000,4000,8000, 7633};

    char files[5][50]={"/home/fas/cpsc424/ahs3/assignment3/C-1000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-2000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-4000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-8000.dat",\
                       "/home/fas/cpsc424/ahs3/assignment3/C-7633.dat"};
    int rank, size;

    // MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int * blocks = (int *) calloc(size, sizeof(int));
    
    if (rank == 0)
        printf("Matrix multiplication times:\n   N      TIME (secs)    F-norm of Error\n -----   -------------  -----------------\n");
    
    for (int run = 4; run < 5; run++) {
        double *A, *B, *C, *Ctrue;
        double wcs, wce, ct;
        FILE *fptr;
        int N = sizes[run];
        int blockSize = N / size;
        int sizeAB = N * (N + 1) / 2;
        int sizeC = N * N;

        // allocate blocks for load blance and generalization
        int avg = (1 + N) * N / (size * 2);
        int index = 0, start = 0, sum = 0;
        for (int i = 1; i <= N; i++) {
            sum += i;
            if (sum >= avg) {
                blocks[index++] = --i - start;
                start = i;
                sum = 0;
            }
            if (index == size - 1) {
                blocks[index] = N - start;
                break;
            }
        }

        if (rank == 0) {
            MPI_Status status; //[2 * size];
            MPI_Request sendRequest[20], sendRequest2[20], sendRequest3[20], recvRequest[20];
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double));
            C = (double *) calloc(sizeC, sizeof(double));
            srand(12345);
            for (int i = 0; i < sizeAB; i++) A[i] = ((double) rand() / (double) RAND_MAX);
            for (int i = 0; i < sizeAB; i++) B[i] = ((double) rand() / (double) RAND_MAX);
            MPI_Barrier(MPI_COMM_WORLD);

            // Send the permanent row to all workers
            timing(&wcs, &ct);

            index = 0;
            for (int i = 1; i < size; i++) {
                index += blocks[i - 1];
                int length = BLOCK_LEN(index, blocks[i]);
                MPI_Isend(A + index * (index + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &sendRequest[i]);
            }
            for(int i = 1; i < size; i++)
                MPI_Wait(&sendRequest[i], &status);            

            // Send column to all workers
            for (int i = 1; i < size; i++) {
                int colIndex = (i) * blockSize;
                int length = BLOCK_LEN(colIndex, blockSize);
                if (i == (size - 1)) length = (colIndex + 1 + N) * (N - colIndex) / 2;
                MPI_Isend(&colIndex, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &sendRequest[i]);
                // MPI_Isend(&length, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &sendRequest3[i]);
                MPI_Isend(B + colIndex * (colIndex + 1) / 2, length, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &sendRequest2[i]);
            }

            for (int i = 1; i < size; i++) {
                MPI_Wait(&sendRequest[i], &status);
                MPI_Wait(&sendRequest2[i], &status);
                //MPI_Wait(&sendRequest3[i], &status);
            }
            // Do the calculation on currant master node
            int rowIndex = 0;
            int colIndex = 0;
            int recvColLen = BLOCK_LEN(colIndex, blockSize);
            int nextColIndex;
            double *col = B;
            double *nextB = (double *) calloc(sizeAB, sizeof(double));
            double *freeNextB = nextB;
            // Send column to next node, receive column from prev node
            for (int i = 1; i < size; i++) {
                MPI_Isend(&colIndex, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &sendRequest[0]);
                MPI_Isend(col, recvColLen, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &sendRequest[1]);
                MPI_Irecv(&nextColIndex, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &recvRequest[0]);
                MPI_Irecv(nextB, sizeAB, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD, &recvRequest[1]);
                int trueBlockSize = blockSize;
                if (BLOCK_LEN(colIndex, blockSize) < recvColLen) trueBlockSize = N - (size - 1) * blockSize;
                block_matmul(A, col, C, rowIndex, colIndex, blocks[0], trueBlockSize, N);
                MPI_Wait(&sendRequest[0], &status);
                MPI_Wait(&sendRequest[1], &status);
                MPI_Wait(&recvRequest[0], &status);
                MPI_Wait(&recvRequest[1], &status);
                MPI_Get_count(&status, MPI_DOUBLE, &recvColLen);
                colIndex = nextColIndex;
                //col = nextB;
                swap(&col, &nextB);
            }

            // Collect results from workers
            index = 0;
            for (int i = 1; i < size; i++) {
                index += blocks[i - 1];        
                MPI_Irecv(C + index * N, blocks[i] * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &recvRequest[i]);
            }
            block_matmul(A, col, C, rowIndex, colIndex, blocks[0], blockSize, N);
            for (int i = 1; i < size; i++)
                MPI_Wait(&recvRequest[i], &status);
            timing(&wce, &ct);
            
            free(A);
            free(B);
            
            Ctrue = (double *) calloc(sizeC, sizeof(double));
            fptr = fopen(files[run], "rb");
            fread(Ctrue, sizeof(double), sizeC, fptr);
            fclose(fptr);

            double Fnorm = 0.;
            for (int i = 0; i < N * N; i++) Fnorm += (Ctrue[i] - C[i]) * (Ctrue[i] - C[i]);
            Fnorm = sqrt(Fnorm);
            printf("  %5d    %9.4f  %15.10f\n", N, wce - wcs, Fnorm);
            free(Ctrue);
            free(C);
            free(freeNextB);
        } else {
            MPI_Status status; //[2 * size];
            MPI_Request sendRequest[20], recvRequest[20];
            int colIndex, nextColIndex, recvColLen;
            int rowIndex = 0;
            for (int i = 0; i < rank; i++) rowIndex += blocks[i];
            A = (double *) calloc(sizeAB, sizeof(double));
            B = (double *) calloc(sizeAB, sizeof(double)); // Most elements containing in the last block
            C = (double *) calloc(sizeC, sizeof(double));
            double *nextB = (double *) calloc(sizeAB, sizeof(double));
            MPI_Barrier(MPI_COMM_WORLD);
            timing(&wcs, &ct);
            MPI_Irecv(A, BLOCK_LEN(rowIndex, blocks[rank]), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &recvRequest[0]);
            MPI_Irecv(&colIndex, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recvRequest[1]);
            MPI_Irecv(B, sizeAB, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &recvRequest[2]);
            MPI_Wait(&recvRequest[0], &status);
            MPI_Wait(&recvRequest[1], &status);
            MPI_Wait(&recvRequest[2], &status);
            MPI_Get_count(&status, MPI_DOUBLE, &recvColLen);
            for (int i = 1; i < size; i++) {
                MPI_Isend(&colIndex, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD, &sendRequest[0]);
                MPI_Isend(B, recvColLen, MPI_DOUBLE, (rank + 1) % size, 1, MPI_COMM_WORLD, &sendRequest[1]);
                MPI_Irecv(&nextColIndex, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &recvRequest[0]);
                MPI_Irecv(nextB, sizeAB, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &recvRequest[1]);
                int trueBlockSize = blockSize;
                if (BLOCK_LEN(colIndex, blockSize) < recvColLen) trueBlockSize = N - (size - 1) * blockSize;
                block_matmul(A, B, C, rowIndex, colIndex, blocks[rank], trueBlockSize, N);                
                MPI_Wait(&recvRequest[0], &status);
                MPI_Wait(&recvRequest[1], &status);
                MPI_Get_count(&status, MPI_DOUBLE, &recvColLen);
                MPI_Wait(&sendRequest[0], &status);
                MPI_Wait(&sendRequest[1], &status);
                colIndex = nextColIndex;
                //B = nextB;
                swap(&B, &nextB);
            }
            int trueBlockSize = blockSize;
            if (BLOCK_LEN(colIndex, blockSize) < recvColLen) trueBlockSize = N - (size - 1) * blockSize;
            block_matmul(A, B, C, rowIndex, colIndex, blocks[rank], trueBlockSize, N);
            timing(&wce, &ct);
            printf("rank = %d process cost %fs for N = %d\n", rank, wce - wcs, N);
            MPI_Send(C, N * blocks[rank], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

            free(A);
            free(B);
            free(C);
            free(nextB);
        }
    }

    MPI_Finalize();
    free(blocks);
}

void block_matmul(double *A, double *B, double *C, int rowIndex, int colIndex, int rowBlockSize, int colBlockSize, int N) {
    int iA, iB, iC;
    for (int i = 0; i < rowBlockSize; i++) {
        iC = i * N + colIndex;
        iA = (2 * rowIndex + i + 1) * i / 2;
        for (int j = 0; j < colBlockSize; j++, iC++) {
            iB = (2 * colIndex + j + 1) * j / 2;
            C[iC] = 0;
            for (int k = 0; k <= MIN(i + rowIndex, j + colIndex); k++) C[iC] += A[iA + k] * B[iB + k];
        }
    }
}

void swap(double** A, double** B) {
    double *tmp = *A;
    *A = *B;
    *B = tmp;
}
