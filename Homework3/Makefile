# NOTE: This Makefile presumes that the following command has been executed:

# module load Langs/Intel/15 MPI/OpenMPI/2.1.1-intel15

TIMINGDIR = /home/fas/cpsc424/ahs3/utils/timing
CC = mpicc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99 -I$(TIMINGDIR)

all: serial task2 task3 task4
task2: task2.o $(TIMINGDIR)/timing.o
task3: task3.o $(TIMINGDIR)/timing.o
task4: task4.o $(TIMINGDIR)/timing.o
serial:	serial.o matmul.o $(TIMINGDIR)/timing.o
	$(CC) -o $@ $(CFLAGS) $^

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial task2 task3 task4 *.o
