CC = gcc
HIPCC = hipcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -std=c23 -D_POSIX_C_SOURCE=199309L
HIPFLAGS = -O2
NVCCFLAGS = -O2 -diag-suppress 550
LDFLAGS = -lm
ARGS ?=

HIP_TARGETS := $(if $(shell which $(HIPCC) 2>/dev/null),matrix_mul_hip)
CUDA_TARGETS := $(if $(shell which $(NVCC) 2>/dev/null),matrix_mul_cuda)

all: matrix_mul_cpu matrix_mul_cpu_transposed matrix_mul_openmp matrix_mul_openmp_transposed $(HIP_TARGETS) $(CUDA_TARGETS)

matrix_mul_cpu: matrix_mul_cpu.c matrix_mul.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

matrix_mul_cpu_transposed: matrix_mul_cpu_transposed.c matrix_mul.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

matrix_mul_openmp: matrix_mul_openmp.c matrix_mul.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

matrix_mul_openmp_transposed: matrix_mul_openmp_transposed.c matrix_mul.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

matrix_mul_hip: matrix_mul_hip.cpp matrix_mul.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

matrix_mul_cuda: matrix_mul_cuda.cu matrix_mul.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-cpu: matrix_mul_cpu
	./matrix_mul_cpu $(ARGS)

run-cpu-transposed: matrix_mul_cpu_transposed
	./matrix_mul_cpu_transposed $(ARGS)

run-openmp: matrix_mul_openmp
	./matrix_mul_openmp $(ARGS)

run-openmp-transposed: matrix_mul_openmp_transposed
	./matrix_mul_openmp_transposed $(ARGS)

run-hip: matrix_mul_hip
	./matrix_mul_hip $(ARGS)

run-cuda: matrix_mul_cuda
	./matrix_mul_cuda $(ARGS)

clean:
	rm -f matrix_mul_cpu matrix_mul_cpu_transposed matrix_mul_openmp matrix_mul_openmp_transposed matrix_mul_hip matrix_mul_cuda

.PHONY: all run-cpu run-cpu-transposed run-openmp run-openmp-transposed run-hip run-cuda clean
