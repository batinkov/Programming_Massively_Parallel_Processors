CC = gcc
HIPCC = hipcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -std=c23 -D_POSIX_C_SOURCE=199309L
HIPFLAGS = -O2
NVCCFLAGS = -O2
LDFLAGS =
ARGS =

# GPU targets are only included if their compilers are available
HIP_TARGETS := $(if $(shell which $(HIPCC) 2>/dev/null),vec_add_hip vec_add_hip_unified)
CUDA_TARGETS := $(if $(shell which $(NVCC) 2>/dev/null),vec_add_cuda)

all: vec_add_cpu vec_add_openmp $(HIP_TARGETS) $(CUDA_TARGETS) test_vec_add

vec_add_cpu: vec_add_cpu.c vec_add.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

vec_add_openmp: vec_add_openmp.c vec_add.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

vec_add_hip: vec_add_hip.cpp vec_add.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

vec_add_hip_unified: vec_add_hip_unified.cpp vec_add.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

vec_add_cuda: vec_add_cuda.cu vec_add.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

test_vec_add: test_vec_add.c vec_add.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run-cpu: vec_add_cpu
	./vec_add_cpu $(ARGS)

run-openmp: vec_add_openmp
	./vec_add_openmp $(ARGS)

run-hip: vec_add_hip
	./vec_add_hip $(ARGS)

run-hip-unified: vec_add_hip_unified
	./vec_add_hip_unified $(ARGS)

run-cuda: vec_add_cuda
	./vec_add_cuda $(ARGS)

test: test_vec_add
	./test_vec_add

valgrind-test: test_vec_add
	valgrind --leak-check=full --error-exitcode=1 ./test_vec_add

valgrind-cpu: vec_add_cpu
	valgrind --leak-check=full --error-exitcode=1 ./vec_add_cpu -n 1000000 -r 2

valgrind-openmp: vec_add_openmp
	valgrind --leak-check=full --error-exitcode=1 --suppressions=openmp.supp ./vec_add_openmp -n 1000000 -r 2

clean:
	rm -f vec_add_cpu vec_add_openmp vec_add_hip vec_add_hip_unified vec_add_cuda test_vec_add

.PHONY: all run-cpu run-openmp run-hip run-hip-unified run-cuda test valgrind-test valgrind-cpu valgrind-openmp clean
