CC = gcc
HIPCC = hipcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -std=c23 -D_POSIX_C_SOURCE=199309L
HIPFLAGS = -O2
NVCCFLAGS = -O2
LDFLAGS = -lm
ARGS ?=

HIP_TARGETS := $(if $(shell which $(HIPCC) 2>/dev/null),blur_hip)

all: blur_cpu blur_openmp $(HIP_TARGETS)

blur_cpu: blur_cpu.c blur.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

blur_openmp: blur_openmp.c blur.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

blur_hip: blur_hip.cpp blur.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

run-cpu: blur_cpu
	./blur_cpu $(ARGS)

run-openmp: blur_openmp
	./blur_openmp $(ARGS)

run-hip: blur_hip
	./blur_hip $(ARGS)

clean:
	rm -f blur_cpu blur_openmp blur_hip

.PHONY: all run-cpu run-openmp run-hip clean
