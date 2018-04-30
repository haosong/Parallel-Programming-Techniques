# This Makefile assumes the following module files are loaded:
#
# Langs/Intel/15
# GPU/Cuda/8.0
#
# This Makefile will only work if executed on a GPU node.
#

TIMINGDIR = /home/fas/cpsc424/ahs3/utils/timing

CC = icc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99 -I$(TIMINGDIR)

CUDAPATH = /gpfs/apps/hpc/GPU/Cuda/8.0

NVCC = $(CUDAPATH)/bin/nvcc
NVCCFLAGS = -I$(CUDAPATH)/include -O3

LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

# Compiler-specific flags (by default, we always use sm_35)
GENCODE_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE = $(GENCODE_SM35)

.SUFFIXES : .cu .ptx

BINARIES = matmul serial task1 task2 task3

all: $(BINARIES)

serial:	serial.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^

task1: task1.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

task2: task2.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

task3: task3.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

matmul: matmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)
