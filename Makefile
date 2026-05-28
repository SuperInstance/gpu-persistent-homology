NVCC = /usr/local/cuda-12.6/bin/nvcc
ARCH = -arch=sm_89
NVCCFLAGS = $(ARCH) -O3 -std=c++17 --extended-lambda -Iinclude
LDFLAGS = -L/usr/lib/wsl/lib -lcuda -L/usr/local/cuda-12.6/lib64 -lcudart

all: test
test: test_correctness
	./test_correctness

SRCS = tests/test_correctness.cu src/distance_matrix.cu src/union_find.cu src/h0_persistence.cu src/h1_persistence.cu src/wasserstein.cu

test_correctness: $(SRCS) include/persistent_homology.cuh
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f test_correctness

.PHONY: all test clean
