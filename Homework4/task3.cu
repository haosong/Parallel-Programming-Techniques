#define FP float
#define TW 3
#define NTB 2

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)  {
    fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    // if (abort) exit(code);
  }
}


__global__ void gpu_matrixmult(FP *a, FP *b, FP *c, int n, int m, int p, int TW1, int NTB1) {

  extern __shared__ FP bigarray[];
  FP *atile = &bigarray[0], *btile = &bigarray[TW * TW], *cvalue = &bigarray[(NTB + 1) * TW * TW];
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = ty + TW * blockIdx.y;

  for (int kt = 0; kt < NTB; kt++) cvalue[kt * TW * TW + ty * TW + tx] = 0.;

  for (int k = 0; k < (TW + p - 1)/TW; k++) {
    atile[ty * TW + tx] = (k*TW + tx < p && row < n) ? a[row*p + k*TW + tx] : 0.;
    for (int kt = 0; kt < NTB; kt++) {
      int col = tx + TW * (blockIdx.x + kt);
      btile[ty * TW + tx] = (k*TW + ty < p && col < m) ? b[(k*TW + ty)*m + col] : 0.;
      __syncthreads();
      for (int i = 0; i < TW; ++i) cvalue[kt * TW * TW + ty * TW + tx] += atile[ty * TW + i] * btile[i * TW + tx];
      __syncthreads();
    }
  }

  for(int kt = 0; kt < NTB; kt++) {
    int col = tx + TW * (blockIdx.x + kt);
    if (row < n && col < m) c[row * m + col] = cvalue[kt * TW * TW + ty * TW + tx];
  }

}


void cpu_matrixmult(FP *a,FP *b, FP *c, int n, int m, int p) {

  size_t index, indexa, indexb;
  FP cvalue;
  for(int col=0;col < m; col++)
    for(int row=0;row < n; row++) {
      indexb = col;
      index = row * m + col;
      cvalue = 0.;
      for (indexa = row*p; indexa < (row*p + p); indexa++, indexb+=m) 
        cvalue += a[indexa]*b[indexb];
      c[index] -= cvalue; //NOTE: This calculates the diff between CPU and GPU computations.
    }
}


int main(int argc, char *argv[]) {

  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Grid_Dim_m = 1, Grid_Dim_n = 1; //Grid dimension, x and y, square
  int Block_Dim = 1; //Block dimension, x and y, square

  int n, m, p; // matrix dimension
  FP *a,*b,*c;
  FP *dev_a, *dev_b, *dev_c;
  size_t size_a, size_b, size_c;

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  // --------------------SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
     printf("Device count = %d\n",gpucount);
  }

  if (argc!=6) {
    printf("Usage: matmul <matrix dim: n> <matrix dim: m> <matrix dim: p> <block dim> <adjacent tiles>\n");
    exit (-1);
  }

  n = atoi(argv[1]);
  m = atoi(argv[2]);
  p = atoi(argv[3]);
  //int NTB = atoi(argv[5]);

  Block_Dim = atoi(argv[4]); // Square block
  // int TW = Block_Dim;
  if (Block_Dim*Block_Dim > 1024) {
    printf("Error, too many threads in block\n");
    exit (-1);
  }

  Grid_Dim_m = (m - 1) / Block_Dim + 1;
  Grid_Dim_n = (n - 1) / Block_Dim + 1;

  cudaSetDevice(gpunum);
  printf("Using device %d\n",gpunum);
  
  printf("Matrix Dimension = [%d, %d, %d]\n",n,m,p);
  printf("Block_Dim = %d, Grid_Dim[m, n] = [%d, %d]\n",Block_Dim,Grid_Dim_m,Grid_Dim_n);

  dim3 Grid(Grid_Dim_m, Grid_Dim_n); //Grid structure
  dim3 Block(Block_Dim, Block_Dim); //Block structure
  
  size_a = n * p * sizeof(FP);
  size_b = p * m * sizeof(FP);
  size_c = n * m * sizeof(FP);
  a = (FP *) malloc(size_a); // dynamically allocated memory for arrays on host
  b = (FP *) malloc(size_b);
  c = (FP *) malloc(size_c); // results from GPU
  printf("size_a = %zu\n", size_a);
  printf("size_b = %zu\n", size_b);
  printf("size_c = %zu\n", size_c);

  srand(12345);
  // int p = n; //Used here only to illustrate proper initialization for non-square case
  printf("a = \n");
  for(i=0;i < n;i++) {
    for(j=0;j < p;j++) {
      //a[i * p + j] = (FP) rand() / (FP) RAND_MAX;
      a[i * p + j] = (FP) i+j; // may be helpful for debugging
      printf("%.5f\t", a[i * p + j]); 
    }
     printf("\n");
  }

  printf("b = \n");
  for(i=0;i < p;i++) {
    for(j=0;j < m;j++) {
      //b[i * m + j] = (FP) rand() / (FP) RAND_MAX;
       b[i * m + j] = (FP) i+j; // may be helpful for debugging
       printf("%.5f\t", b[i * m + j]);
    }
     printf("\n");
  }
  
  for(i=0;i < n;i++)
    for(j=0;j < m;j++)
      c[i * m + j] = 0.;

  // ------------- COMPUTATION DONE ON GPU ----------------------------

  gpuErrchk(cudaMalloc((void**)&dev_a, size_a)); // allocate memory on device
  gpuErrchk(cudaMalloc((void**)&dev_b, size_b));
  gpuErrchk(cudaMalloc((void**)&dev_c, size_c));

  gpuErrchk(cudaMemcpy(dev_a, a , size_a ,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dev_b, b , size_b ,cudaMemcpyHostToDevice));

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);
  // cudaEventSynchronize(start); // not needed

  size_t Ns = ((2 * NTB + 1) * TW * TW) * sizeof(FP);
  gpu_matrixmult<<<Grid,Block,Ns>>>(dev_a,dev_b,dev_c,n,m,p,TW, NTB);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );

  gpuErrchk(cudaMemcpy(c,dev_c, size_c ,cudaMemcpyDeviceToHost));

  printf("c0 = \n");
  for(i=0;i < n;i++) {
    for(j=0;j < m;j++) {
        printf("%.5f\t", c[i * m + j]);
    }
    printf("\n");
  }

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time

  // ------------- COMPUTATION DONE ON HOST CPU ----------------------------
  // DEBUGGING USE ONLY (AND FOR LIMITED NUMBERS OF TIMING RUNS)

  cudaEventRecord(start, 0); // use same timing
  // cudaEventSynchronize(start); // not needed


  cpu_matrixmult(a,b,c, n,m,p); // do calculation on host (NOTE: This computes the diff with GPU result.)

  printf("c1 = \n");
  for(i=0;i < n;i++) {
    for(j=0;j < m;j++) {
        printf("%.5f\t", c[i * m + j]);
    }
    printf("\n");
  }

  cudaEventRecord(stop, 0); // instrument code to measue end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );

  printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms); // exec. time

// ------------------- check device creates correct results -----------------
  double error, suma, sumb, sumc, ai, bi, ci;
  suma = 0.; sumb = 0; sumc = 0;
  for(size_t i=0;i<n*p;i++) {
    ai = (double) a[i];
    suma += ai * ai;
  }
  for(size_t i=0;i<p*m;i++) {
    bi = (double) b[i];
    sumb += bi * bi;
  }
  for(size_t i=0;i<n*m;i++) {
    ci = (double) c[i];
    sumc += ci*ci;
  }
  suma = sqrt(suma);
  sumb = sqrt(sumb);
  sumc = sqrt(sumc);
  error = sumc/(sqrt(n*m)*suma*sumb);
  printf("Scaled error between GPU and CPU: %e\n", error);

// -------------- clean up ---------------------------------------

  free(a);
  free(b);
  free(c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
