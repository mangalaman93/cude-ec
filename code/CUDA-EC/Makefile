EXECUTABLE=CUDA-EC
CUFILES=FixErrorsVoting.cu 
CUDEPS=FixErrorsVoting_kernel.cu
CCFILES=hash_function.cpp utils.cpp main.cpp

#To resolve dependency with cutil.h
OLDCUDALIBS=./common/lib
OLDCUDAINCS=./common/inc
	

################################################################################
# Locate nvcc 
NVCC=nvcc
ARCH=-arch=sm_20
#define ARGS for command line debug 

all: cudacode.o  $(CCFILES)
	make -C ./common
	$(NVCC) $(ARCH) $(ARGS) -I$(OLDCUDAINCS) -L$(OLDCUDALIBS) -lcutil_x86_64 $(CCFILES) cudacode.o -o  $(EXECUTABLE)

cudacode.o: $(CUFILES) $(CUDEPS)
	$(NVCC) $(ARCH) $(ARGS) -I$(OLDCUDAINCS) -c $(CUFILES) -o cudacode.o

clean:
	rm -f $(EXECUTABLE)
	rm -f *.o
	make clean -C ./common

check:
	diff output/SRR006331.fasta.fixed output/SRR006331.fasta.fixed.master | wc -l 
