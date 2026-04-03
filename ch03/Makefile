CC = gcc
HIPCC = hipcc
CFLAGS = -Wall -Wextra -O2 -std=c23 -D_POSIX_C_SOURCE=199309L
HIPFLAGS = -O2
LDFLAGS = -lm
ARGS ?=

# GPU targets are only included if their compilers are available
HIP_TARGETS := $(if $(shell which $(HIPCC) 2>/dev/null),grayscale_hip)

all: grayscale_cpu grayscale_openmp $(HIP_TARGETS)

grayscale_cpu: grayscale_cpu.c grayscale.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

grayscale_openmp: grayscale_openmp.c grayscale.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

grayscale_hip: grayscale_hip.cpp grayscale.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

run-cpu: grayscale_cpu
	./grayscale_cpu $(ARGS)

run-openmp: grayscale_openmp
	./grayscale_openmp $(ARGS)

run-hip: grayscale_hip
	./grayscale_hip $(ARGS)

clean:
	rm -f grayscale_cpu grayscale_openmp grayscale_hip

.PHONY: all run-cpu run-openmp run-hip clean
