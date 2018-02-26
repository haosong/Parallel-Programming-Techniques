CC = icc
CFLAGS = -g -O3 -Wall -xHost -fno-alias -std=c99 -openmp
EXECUTABLES = task1 task2_2 task2_3 task2_4 task3_1 task3_2 task3_3 task4

all: $(EXECUTABLES)

task1: task1.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task2_2: task2_2.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task2_3: task2_3.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task2_4: task2_4.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task3_1: task3_1.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task3_2: task3_2.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task3_3: task3_3.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

task4: task4.o drand.o timing.o
	$(CC) $(CFLAGS) -o $@ $^

drand.o: drand.c
	$(CC) -c $^ $(CFLAGS) -o $@

task1.o: task1.c
	$(CC) -c $^ $(CFLAGS) -o $@

task2_2.o: task2_2.c
	$(CC) -c $^ $(CFLAGS) -o $@

task2_3.o: task2_3.c
	$(CC) -c $^ $(CFLAGS) -o $@

task2_4.o: task2_4.c
	$(CC) -c $^ $(CFLAGS) -o $@

task3_1.o: task3_1.c
	$(CC) -c $^ $(CFLAGS) -o $@

task3_2.o: task3_2.c
	$(CC) -c $^ $(CFLAGS) -o $@

task3_3.o: task3_3.c
	$(CC) -c $^ $(CFLAGS) -o $@

task4.o: task4.c
	$(CC) -c $^ $(CFLAGS) -o $@

clean:
	rm -f $(EXECUTABLES)
	rm -f drand.o task1.o task2_2.o task2_3.o task2_4.o task3_1.o task3_2.o task3_3.o task4.o
