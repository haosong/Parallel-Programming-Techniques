CC = icc
CFLAGS_a = -g -O0 -fno-alias -std=c99
CFLAGS_b = -g -O1 -fno-alias -std=c99
CFLAGS_c = -g -O3 -no-vec -no-simd -fno-alias -std=c99
CFLAGS_d = -g -O3 -xHost -fno-alias -std=c99 #-vec-report

all: exercise1a exercise1b exercise1c exercise1d exercise1_division exercise2

exercise1a: exercise1a.o timing.o
	$(CC) -o $@ $(CFLAGS_a) $^

exercise1b: exercise1b.o timing.o
	$(CC) -o $@ $(CFLAGS_b) $^

exercise1c: exercise1c.o timing.o
	$(CC) -o $@ $(CFLAGS_c) $^

exercise1d: exercise1d.o timing.o
	$(CC) -o $@ $(CFLAGS_d) $^

exercise1_division: exercise1_division.o timing.o
	$(CC) -o $@ $(CFLAGS_a) $^

exercise2: exercise2.o timing.o dummy.o
	$(CC) -o $@ $(CFLAGS_d) $^

exercise1a.o: exercise1.c
	$(CC) -c $^ $(CFLAGS_a) -o $@

exercise1b.o: exercise1.c
	$(CC) -c $^ $(CFLAGS_b) -o $@

exercise1c.o: exercise1.c
	$(CC) -c $^ $(CFLAGS_c) -o $@

exercise1d.o: exercise1.c
	$(CC) -c $^ $(CFLAGS_d) -o $@

exercise1_division.o: exercise1_division.c
	$(CC) -c $^ $(CFLAGS_a) -o $@

exercise2.o: exercise2.c
	$(CC) -c $^ $(CFLAGS_d) -o $@

dummy.o: dummy.c
	$(CC) -c $^ $(CFLAGS_d) -o $@

clean:
	rm -f exercise1a exercise1b exercise1c exercise1d exercise1_division exercise2 *.optrpt
	rm -f exercise1a.o exercise1b.o exercise1c.o exercise1d.o exercise1_division.o exercise2.o dummy.o
