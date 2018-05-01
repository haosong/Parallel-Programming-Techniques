#define FP float

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timing.h"

void cpu_matrixmult(FP *a, FP *b, FP *c, int n, int m, int p) {
    // Same as matrix_multiplication.pdf Page 8, Matrix Multiplication (kij)
    for (int k = 0; k < p; k++) {
        for (int i = 0; i < n; i++) {
            FP r = a[i * p + k];
            for (int j = 0; j < m; j++)
                c[i * m + j] += r * b[k * m + j];
        }
    }
}

int main(int argc, char *argv[]) {

    int i, j; // loop counters
    int n, m, p; // matrix dimension
    FP *a, *b, *c;
    double wcs, wce, ct;

    // --------------------SET PARAMETERS AND DATA -----------------------
    if (argc != 4) {
        printf("Usage: matmul <matrix dim -n> <matrix dim -m> <matrix dim -p>\n");
        exit(-1);
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    p = atoi(argv[3]);
    printf("Matrix Dimension n = %d\n", n);
    printf("Matrix Dimension m = %d\n", m);
    printf("Matrix Dimension p = %d\n", p);

    a = (FP *) malloc(n * p * sizeof(FP)); // dynamically allocated memory for arrays on host
    b = (FP *) malloc(p * m * sizeof(FP));
    c = (FP *) calloc(n * m, sizeof(FP)); // results from CPU

    srand(12345);
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++) {
            a[i * p + j] = (FP) rand() / (FP) RAND_MAX;
            // a[i * p + j] = (FP) i+j; // may be helpful for debugging
        }

    for (i = 0; i < p; i++)
        for (j = 0; j < m; j++) {
            b[i * m + j] = (FP) rand() / (FP) RAND_MAX;
            //b[i * m + j] = (FP) i+j; // may be helpful for debugging
        }
    
    // ------------- COMPUTATION DONE ON HOST CPU ----------------------------
    timing(&wcs, &ct);
    cpu_matrixmult(a, b, c, n, m, p); // do calculation on host (NOTE: This computes the diff with GPU result.)
    timing(&wce, &ct);

    printf("Time to calculate results on CPU: %f ms.\n", (wce - wcs) * 1000); // exec. time

    // -------------- clean up ---------------------------------------
    free(a);
    free(b);
    free(c);

    return 0;
}
