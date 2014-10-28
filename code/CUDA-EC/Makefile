EXECUTABLE=CUDA-EC
CUFILES=FixErrorsVoting.cu
CU_DEPS=FixErrorsVoting_kernel.cu
CCFILES=hash_function.cpp utils.cpp main.cpp

#To resolve dependency with cutil.h
OLDCUDALIBS=./common/lib
OLDCUDAINCS=./common/inc
	

################################################################################
# Locate nvcc 
CUDAROOT=/opt/cuda-4.2/cuda
NVCC=$(CUDAROOT)/bin/nvcc

all: cudacode.o 
	make -C ./common
	$(NVCC) -I$(OLDCUDAINCS) -L$(OLDCUDALIBS) -lcutil_x86_64 $(CCFILES) cudacode.o -o  $(EXECUTABLE)

cudacode.o:
	$(NVCC) -I$(OLDCUDAINCS) -c $(CUFILES) -o cudacode.o

clean:
	rm -f $(EXECUTABLE)
	rm -f *.o
	make clean -C ./common
