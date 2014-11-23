/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */


// includes, system
#include <stdlib.h>
#include <stdio.h>

// includes, project
#include <cutil.h>
#include "FixErrorsVoting.h"
#include "utils.h"

// includes, kernels
#include "FixErrorsVoting_kernel.cu"

//Error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
extern "C" void runTest(unsigned char *hash_table,
						Param *h_param,
						char *reads_arr,
						unsigned int table_size,
						FILE *seqOut,
						FILE *discardOut,
						int argc,
						char** argv)
{
  // Cuda init
  CUT_DEVICE_INIT(argc,argv);

	//texture, linear memory
	// part1, allocate data on device
	unsigned char *dev_hash;
	cudaMalloc( (void**)&dev_hash, table_size );
	// part2, copy memory to device
  cudaMemcpy( dev_hash, hash_table, table_size, cudaMemcpyHostToDevice );
	// part2a, bind texture
  cudaBindTexture(0, tex, dev_hash );	
	printf("Bind texture done..\n");	
	
	char *d_reads_arr;	
	Param *d_param;	
	
	//CALL KERNEL MULTIPLE TIME

	unsigned int timer = 0;
	double totaltime=0.0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	//allocate memory on Device
	gpuErrchk( cudaMalloc((void**) &d_reads_arr, sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS) );
	gpuErrchk( cudaMalloc((void**) &d_param, sizeof(Param)) );
				
	printf( "Allocate memory on device done...\n");			
	
	//copy from CPU to GPU
	gpuErrchk( cudaMemcpy(d_reads_arr,reads_arr,sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS,cudaMemcpyHostToDevice) );	
	gpuErrchk( cudaMemcpy(d_param,h_param,sizeof(Param),cudaMemcpyHostToDevice) );	
		
	printf( "Copy from CPU to GPU done...\n");


	dim3 Block_dim(BLOCK,1); //N block, each block has M thread
	dim3 Thread_dim(THREAD,1);

	//call kernel
	printf( "Running Kernel with %d Block, %d Thread...\n",BLOCK,THREAD);

  //Given number of bytes required for shared memory
	fix_errors1<<<Block_dim,Thread_dim,(h_param->readLen + 2)*THREAD>>>(d_reads_arr,d_param);

  gpuErrchk( cudaPeekAtLastError() );
	CUT_SAFE_CALL(cutStopTimer(timer));
	totaltime = cutGetTimerValue(timer);

	//copy from GPU to CPU
	gpuErrchk( cudaMemcpy(reads_arr,d_reads_arr, sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS, cudaMemcpyDeviceToHost) );	
		
	printf("GPU pure time: %f msec\n", totaltime);
	printf( "Copy from GPU to CPU done...\n");


	if(h_param->numSearch == 2)
	{
		///////////////////////////////////////////////// Fix two errors /////////////////////////////

		CUT_SAFE_CALL( cutStartTimer( timer));

		//get the rest un-fixed reads
		int i, numUnfixed=0;

		for(i=0;i<h_param->NUM_OF_READS;i++){
			if(reads_arr[i*(h_param->readLen + 2)+h_param->readLen]== 'D'){
				numUnfixed ++;
			}
		}

		char *reads_array_unfixed =(char *)malloc(sizeof(char)*numUnfixed*(h_param->readLen + 2));

		numUnfixed = 0;
		for(i=0;i<h_param->NUM_OF_READS;i++){
			if(reads_arr[i*(h_param->readLen + 2)+h_param->readLen]== 'D'){
				strncpy(&reads_array_unfixed[numUnfixed*(h_param->readLen + 2)], &reads_arr[i*(h_param->readLen + 2)], (h_param->readLen + 2));

				numUnfixed++;
			}
		}

		printf( "Total number of un-fixed reads after 1st kernel is %d...\n",numUnfixed);

		//allocate memory on Device
		CUDA_SAFE_CALL(cudaFree(d_reads_arr));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_reads_arr, sizeof(char)*(h_param->readLen + 2)*numUnfixed));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_param, sizeof(Param)));

		printf( "Allocate memory on device done...\n");

		//copy from CPU to GPU
		CUDA_SAFE_CALL(cudaMemcpy(d_reads_arr,reads_array_unfixed,sizeof(char)*(h_param->readLen + 2)*numUnfixed,cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_param,h_param,sizeof(Param),cudaMemcpyHostToDevice));

		printf( "Copy from CPU to GPU done...\n");
		printf( "Running Kernel with %d Block, %d Thread...\n",BLOCK,THREAD);

		//fix error number 2
		fix_errors2<<<Block_dim,Thread_dim>>>(d_reads_arr,d_param,numUnfixed);		
    gpuErrchk( cudaPeekAtLastError() );
			
		CUT_SAFE_CALL(cutStopTimer(timer));
		totaltime = cutGetTimerValue(timer);

		//copy from GPU to CPU
		CUDA_SAFE_CALL(cudaMemcpy(reads_array_unfixed,d_reads_arr, sizeof(char)*(h_param->readLen + 2)*numUnfixed, cudaMemcpyDeviceToHost));

		printf("GPU pure time: %f msec\n", totaltime);
		printf( "Copy from GPU to CPU done...\n");

		//consolidate
		numUnfixed = 0;
		for(i=0;i<h_param->NUM_OF_READS;i++)
		{
			if(reads_arr[i*(h_param->readLen + 2)+h_param->readLen]== 'D')
			{
				strncpy(&reads_arr[i*(h_param->readLen + 2)],&reads_array_unfixed[numUnfixed*(h_param->readLen + 2)], (h_param->readLen + 2));
				numUnfixed++;
			}
		}
		free(reads_array_unfixed);
	}

	printf("...releasing GPU memory.\n");
    cudaUnbindTexture(tex);
    CUDA_SAFE_CALL(cudaFree(dev_hash));

    CUDA_SAFE_CALL(cudaFree(d_reads_arr));
	CUDA_SAFE_CALL(cudaFree(d_param));

	CUT_SAFE_CALL(cutDeleteTimer(timer));
}

