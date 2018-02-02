#!/bin/bash

echo -e "\n====== Make ======\n"
make clean
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb make

echo -e "\n\n====== Run Exercise 1 with Compiler Option (a) ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise1a

echo -e "\n====== Run Exercise 1 with Compiler Option (b) ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise1b

echo -e "\n====== Run Exercise 1 with Compiler Option (c) ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise1c

echo -e "\n====== Run Exercise 1 with Compiler Option (d) ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise1d

echo -e "\n====== Run Exercise 1 Division Operation Latency Test ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise1_division

echo -e "\n====== Run Exercise 2 ======"
srun --pty -p interactive -c 1 -t 6:00:00 --mem-per-cpu=6100mb ./exercise2

echo -e "\n\n"

