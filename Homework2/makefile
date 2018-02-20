CC = icc
CFLAGS = -g -O3 -xHost -fno-alias -std=c99

# CC = gcc
# CFLAGS = -g -O3 -std=c99

all: task1 #task2 task3

task1: task1.o drand.o timing.o
# task2: task2.o timing.o
# task3: task3.o timing.o
	$(CC) -o $@ $(CFLAGS) $^

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f task1 task1.o drand.o
	# rm -f task2 task2.o
	# rm -f task3 task3.o
