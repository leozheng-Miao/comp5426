# Makefile for compiling and running MPI programs

# Compiler
CC = mpicc

# Compiler flags
CFLAGS = -O3

# Executable names
EXE1 = withoutLoop
EXE2 = withLoop

# Source files
SRC1 = withoutLoop.c
SRC2 = withLoop.c

# Default target
all: $(EXE1) $(EXE2)

# Compile the first program
$(EXE1): $(SRC1)
	$(CC) $(CFLAGS) -o $(EXE1) $(SRC1)

# Compile the second program
$(EXE2): $(SRC2)
	$(CC) $(CFLAGS) -o $(EXE2) $(SRC2)

# Clean up the executables
clean:
	rm -f $(EXE1) $(EXE2)

# Run the first program with default arguments
run1:
	mpirun -np 2 ./$(EXE1) 128 2

# Run the second program with default arguments
run2:
	mpirun -np 2 ./$(EXE2) 128

.PHONY: all clean run1 run2

