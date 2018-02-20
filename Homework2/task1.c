#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "drand.h"
#include "timing.h"

int main(int argc, char const *argv[]) {
    dsrand(12345);
    // complex a = I;
    // printf("%f\n", cabs(a));
    double wcs, wce, ct;
    timing(&wcs, &ct);
    double x = -2.0, y = 0.0;
    int N0 = 0, N1 = 0;
    for (double x = -2.0; x < 0.5; x += 0.001) {
        for (double y = 0.0; y < 1.25; y += 0.001) {
            double real = x + drand() * 0.001;
            double imag = (y + drand() * 0.001) * I;
            double complex zz = 1.0 + 1.0*I;
            double complex c = real + imag;
            double complex z = c;
            printf("The conjugate of Z1 = %.2f %+.2fi\n", creal(c), cimag(c));
            printf("abs of c = %.2f\n", cabs(c));
            int i = 0;
            while (i++ < 20000) {
                // double complex temp = z * z;
                // temp = temp + c;
                // if (cabs(c) > 2) {
                    // N0++;
                    // break;
                // }
            }
            if (i == 20000) N1++;
        }
    }
    timing(&wce, &ct);
    double runtime = wce - wcs;
    double area = 2 * N1 * 3.125 / (N1 + N0);
    printf("%f\n", area);
    printf("%f\n", runtime);
    return 0;
}