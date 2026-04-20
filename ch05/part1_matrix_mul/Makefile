CC = gcc
HIPCC = hipcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -std=c23 -D_POSIX_C_SOURCE=199309L
HIPFLAGS = -O2
NVCCFLAGS = -O2 -diag-suppress 550
LDFLAGS = -lm
ARGS ?=

HIP_TARGETS := $(if $(shell which $(HIPCC) 2>/dev/null),matrix_mul_hip_tiled)
CUDA_TARGETS := $(if $(shell which $(NVCC) 2>/dev/null),matrix_mul_cuda_tiled)

all: $(HIP_TARGETS) $(CUDA_TARGETS)

matrix_mul_hip_tiled: matrix_mul_hip_tiled.cpp matrix_mul.h
	$(HIPCC) $(HIPFLAGS) -o $@ $<

matrix_mul_cuda_tiled: matrix_mul_cuda_tiled.cu matrix_mul.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-hip-tiled: matrix_mul_hip_tiled
	./matrix_mul_hip_tiled $(ARGS)

run-cuda-tiled: matrix_mul_cuda_tiled
	./matrix_mul_cuda_tiled $(ARGS)

clean:
	rm -f matrix_mul_hip_tiled matrix_mul_cuda_tiled

.PHONY: all run-hip-tiled run-cuda-tiled clean
