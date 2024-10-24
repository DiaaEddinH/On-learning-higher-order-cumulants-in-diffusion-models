# Compiler
CC = gcc

# Flags
CFLAGS = -Wall -Wextra -I. #-Xclang -fopenmp

# Target
TARGET = test_main #mpi_main

# Source files
SRCS = test_main.c cumulants.c

# Object files
OBJS = $(SRCS:.c=.o)

# Build target
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS)

#Clean up build files
clean:
	rm -f $(OBJS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
