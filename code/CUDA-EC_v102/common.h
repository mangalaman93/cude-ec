FixErrorsVoting.cu                                                                                  0000664 0000765 0000765 00000015574 11167253302 014713  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               /*
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
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_reads_arr, sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_param, sizeof(Param)));
				
	printf( "Allocate memory on device done...\n");			
	
	//copy from CPU to GPU
	CUDA_SAFE_CALL(cudaMemcpy(d_reads_arr,reads_arr,sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS,cudaMemcpyHostToDevice));	
	CUDA_SAFE_CALL(cudaMemcpy(d_param,h_param,sizeof(Param),cudaMemcpyHostToDevice));	
		
	printf( "Copy from CPU to GPU done...\n");

	
	dim3 Block_dim(BLOCK,1); //N block, each block has M thread
	dim3 Thread_dim(THREAD,1);
		
	//call kernel
	printf( "Running Kernel with %d Block, %d Thread...\n",BLOCK,THREAD);
		
	fix_errors1<<<Block_dim,Thread_dim>>>(d_reads_arr,d_param);
		
	CUT_CHECK_ERROR("Kernel execution failed");
		
	CUT_SAFE_CALL(cutStopTimer(timer));
	totaltime = cutGetTimerValue(timer);
		
	//copy from GPU to CPU
	CUDA_SAFE_CALL(cudaMemcpy(reads_arr,d_reads_arr, sizeof(char)*(h_param->readLen + 2)*h_param->NUM_OF_READS, cudaMemcpyDeviceToHost));	
		
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
			
		CUT_CHECK_ERROR("Kernel execution failed");
			
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

                                                                                                                                    FixErrorsVoting.h                                                                                   0000664 0000765 0000765 00000003316 11167253302 014522  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               #ifndef FIXERRORSVOTING_H_
#define FIXERRORSVOTING_H_

#include "hash_function.h"

//Must define read length and tuple size as it used in kernel, which cannot be dynamic allocate memory. 
//In char[READ_LENGTH+1]
#define READ_LENGTH 35
#define TUPLE_SIZE 20

//#define NUM_OF_READS 1153738 // NC000913: 9279350//mw2:3857879 //NC007146:3828980 //NC001137:1153738 //NC001139: 2181894
//helicobacter,11628131

#define NUM_HASH 8 //number of hash function
#define BLOOM_SIZE 8

//Block and Thread number used in Kernel
#define BLOCK 256
#define THREAD 256

#define MAX_READS_BOUND BLOCK * THREAD  //65536 //131072 //65536

#define	NAMESTR_LEN 60

typedef struct {
	short length;
	//int  _masked;
	//int _ascii;		
	//int fixed;
	char namestr[NAMESTR_LEN];
	char seq[READ_LENGTH+1];
}DNASequence;


typedef struct{	 
	//char *spectrumType; 
	short doTrim; 
	short doDeletion;
	short doInsertion; 
	short maxMods; 
	short minVotes; 
	short maxTrim;
	int   numTuples;
	short numSearch;	
	short tupleSize;
	int   NUM_OF_READS;
	short readLen;
}Param;

const unsigned int char_size = 0x08;    // 8 bits in 1 char(unsigned)

const unsigned char bit_mask[8] = {
                                   0x01, //00000001
                                   0x02, //00000010
                                   0x04, //00000100
                                   0x08, //00001000
                                   0x10, //00010000
                                   0x20, //00100000
                                   0x40, //01000000
                                   0x80  //10000000
                                 };

typedef unsigned int (*hash_function)(char*, unsigned int len);


#endif
                                                                                                                                                                                                                                                                                                                  FixErrorsVoting_kernel.cu                                                                           0000664 0000765 0000765 00000111531 11167253302 016241  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               /*
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

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _FIXERRORSVOTING_KERNEL_H_
#define _FIXERRORSVOTING_KERNEL_H_

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "FixErrorsVoting.h"

texture<unsigned char,1, cudaReadModeElementType> tex;


__constant__ unsigned int _char_size_ = 0x08;    // 8 bits in 1 char(unsigned)

__constant__ unsigned char _bit_mask_[8] = {
                                   0x01, //00000001
                                   0x02, //00000010
                                   0x04, //00000100
                                   0x08, //00001000
                                   0x10, //00010000
                                   0x20, //00100000
                                   0x40, //01000000
                                   0x80  //10000000
                                 };
                                 
__device__ char nextNuc[256];


__constant__ char unmasked_nuc[256] = {0, 1, 2, 3, 'N', 'R', 'Y', 'W', 'S', 'M',     // 9	
			     'K', 'H', 'B', 'V', 'D', 'X', '\0','\0','\0','\0',    // 19	
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 29 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 39 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 49 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 59 
			     '\0','\0','\0','\0','\0', 'A','\0', 'C','\0','\0',    // 69 
			     '\0', 'G', 'H','\0','\0', 'K','\0', 'M', 'N','\0',    // 79 
			     '\0','\0', 'R', 'S', 'T','\0','\0', 'W', 'X', 'Y',    // 89 
			     '\0','\0','\0','\0','\0','\0','\0', 'A','\0', 'C',    // 99 
			     '\0','\0','\0', 'G', 'H','\0','\0', 'K','\0', 'M',    // 109
			     'N', '\0','\0','\0', 'R', 'S', 'T','\0','\0','\0',    // 119
			     'X', '\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 129
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 139
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 149
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 159
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 169
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 179
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 189
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 199
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 209
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 219
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 229
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 239
			     '\0','\0','\0','\0',   0,   1,   2,   3,   0,   1,    // 249
			        2,   3,   0,   1,   2,   3};                       // 255


__constant__ char nuc_char[256] = {'G', 'A', 'C', 'T', 'N', 'R', 'Y', 'W', 'S', 'M',     // 9	
			     'K', 'H', 'B', 'V', 'D', 'X', '\0','\0','\0','\0',    // 19	
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 29 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 39 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 49 
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 59 
			     '\0','\0','\0','\0','\0', 'A','\0', 'C','\0','\0',    // 69 
			     '\0', 'G', 'H','\0','\0', 'K','\0', 'M', 'N','\0',    // 79 
			     '\0','\0', 'R', 'S', 'T','\0','\0', 'W','\0', 'Y',    // 89 
			     '\0','\0','\0','\0','\0','\0','\0', 'a','\0', 'c',    // 99 
			     '\0','\0','\0', 'g', 'h','\0','\0', 'k','\0', 'm',    // 109
			     'n', '\0','\0','\0', 'r', 's', 't','\0','\0','\0',    // 119
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 129
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 139
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 149
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 159
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 169
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 179
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 189
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 199
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 209
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 219
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 229
			     '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',    // 239
			     '\0','\0','\0','\0', 'g', 'a', 'c', 't', 'g', 'a',    // 249
			     'c', 't','g','a','c','t'};                            // 255

__constant__ char unmasked_nuc_index[256] = {
		  0,1,2,3,4,4,4,4,4,4,   // 0 9
		  4,4,4,4,4,4,4,4,4,4,   // 10 19
		  4,4,4,4,4,4,4,4,4,4,   // 20 29
		  4,4,4,4,4,4,4,4,4,4,   // 30 39
		  4,4,4,4,4,4,4,4,4,4,   // 40 49
		  4,4,4,4,4,4,4,4,4,4,   // 50 59
		  4,4,4,4,4,1,4,2,4,4,   // 60 69
		  4,0,4,4,4,4,4,4,4,4,   // 70 79
		  4,4,4,4,3,4,4,4,4,4,   // 80 89
		  4,4,4,4,4,4,4,1,4,2, // 90 99
		  4,4,4,0,4,4,4,4,4,4,  // 100 109
		  4,4,4,4,4,4,3,4,4,4,  // 110 119
		  4,4,4,4,4,4,4,4,4,4,   // 120 129
		  4,4,4,4,4,4,4,4,4,4,   // 130 139
		  4,4,4,4,4,4,4,4,4,4,   // 140 149
		  4,4,4,4,4,4,4,4,4,4,   // 150 159
		  4,4,4,4,4,4,4,4,4,4,   // 160 169
		  4,4,4,4,4,4,4,4,4,4,   // 170 179
		  4,4,4,4,4,4,4,4,4,4,   // 180 189
		  4,4,4,4,4,4,4,4,4,4,   // 190 199
		  4,4,4,4,4,4,4,4,4,4,   // 200 209
		  4,4,4,4,4,4,4,4,4,4,   // 210 219
		  4,4,4,4,4,4,4,4,4,4,   // 220 229
		  4,4,4,4,4,4,4,4,4,4,   // 230 239
		  4,4,4,4,0,1,2,3,0,1,   // 240 249
		  2,3,0,1,2,3 };   // 250 255

__constant__ char numeric_nuc_index[256] = {
		  0,1,2,3,4,4,4,4,4,4,   // 0 9
		  4,4,4,4,4,4,4,4,4,4,   // 10 19
		  4,4,4,4,4,4,4,4,4,4,   // 20 29
		  4,4,4,4,4,4,4,4,4,4,   // 30 39
		  4,4,4,4,4,4,4,4,4,4,   // 40 49
		  4,4,4,4,4,4,4,4,4,4,   // 50 59
		  4,4,4,4,4,1,4,2,4,4,   // 60 69
		  4,0,4,4,4,4,4,4,4,4,   // 70 79
		  4,4,4,4,3,4,4,4,4,4,   // 80 89
		  4,4,4,4,4,4,4,-3,4,-2, // 90 99
		  4,4,4,-4,4,4,4,4,4,4,  // 100 109
		  4,4,4,4,4,4,-1,4,4,4,  // 110 119
		  4,4,4,4,4,4,4,4,4,4,   // 120 129
		  4,4,4,4,4,4,4,4,4,4,   // 130 139
		  4,4,4,4,4,4,4,4,4,4,   // 140 149
		  4,4,4,4,4,4,4,4,4,4,   // 150 159
		  4,4,4,4,4,4,4,4,4,4,   // 160 169
		  4,4,4,4,4,4,4,4,4,4,   // 170 179
		  4,4,4,4,4,4,4,4,4,4,   // 180 189
		  4,4,4,4,4,4,4,4,4,4,   // 190 199
		  4,4,4,4,4,4,4,4,4,4,   // 200 209
		  4,4,4,4,4,4,4,4,4,4,   // 210 219
		  4,4,4,4,4,4,4,4,4,4,   // 220 229
		  4,4,4,4,4,4,4,4,4,4,   // 230 239
		  4,4,4,4,-12,-11,-10,-9,-8,-7,   // 240 249
		  -6,-5,-4,-3,-2,-1 };   // 250 255

 __constant__ unsigned char nucToIndex[256] = {16,16,16,16,16,16,16,16,16,16,  // 0
				      16,16,16,16,16,16,16,16,16,16,  // 10
				      16,16,16,16,16,16,16,16,16,16,  // 20
				      16,16,16,16,16,16,16,16,16,16,  // 30
				      16,16,16,16,16,16,16,16,16,16,  // 40
				      16,16,16,16,16,16,16,16,16,16,  // 50
				      16,16,16,16,16,1,12,2,14,16,  // 60
				      16,0,11,16,16,116,16,9,4,16,  // 70
				      16,16,5,8,3,16,13,7,15,6,  // 80
				      16,16,16,16,16,16,16,253,12,254,  // 90
				      14,16,16,252,11,8,16,116,16,9,  // 100
				      4,16,16,16,5,16,255,16,13,7,  // 110
				      15,6,16,16,16,16,16,16,16,16,  // 120
				      16,16,16,16,16,16,16,16,16,16,  // 130
				      16,16,16,16,16,16,16,16,16,16,  // 140
				      16,16,16,16,16,16,16,16,16,16,  // 150
				      16,16,16,16,16,16,16,16,16,16,  // 160
				      16,16,16,16,16,16,16,16,16,16,  // 170
				      16,16,16,16,16,16,16,16,16,16,  // 180
				      16,16,16,16,16,16,16,16,16,16,  // 190
				      16,16,16,16,16,16,16,16,16,16,  // 200
				      16,16,16,16,16,16,16,16,16,16,  // 210
				      16,16,16,16,16,16,16,16,16,16,  // 220
				      16,16,16,16,16,16,16,16,16,16,  // 230
				      16,16,16,16,16,16,16,16,16,16,  // 240
				      16,16,16,16,16,16};         // 250

  
  __constant__ char indexToNuc[256] = {'G', 'A', 'C', 'T', 'N', 'R', 'Y', 'W', 'S', 'M',   // 9
				      'K', 'H', 'B', 'V', 'D', 'X', '\0','\0','\0','\0',  // 19
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 29 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 39 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 49 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 59 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 69 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 79 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 89 
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 99
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 109
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 119
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 129
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 139
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 149
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 159
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 169
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 179
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 189
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 199
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 209
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 219
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 229
				      '\0','\0','\0','\0','\0','\0','\0','\0','\0','\0',  // 239
				      '\0','\0','\0','\0', 'g', 'a', 'c', 't', 'g', 'a',  // 249
				      'c', 't','g','a','c','t'};                          // 255


	
__device__ int _toupper_(int ch) 
{
  if ((unsigned int)(ch - 'a') < 26u )
    ch += 'A' - 'a';
  return ch;
}

__device__ char * _strcpy_(char *s1, char *s2)
{
	char *os1;

	os1 = s1;
	while (*s1++ = *s2++)
		;
	return(os1);
}


__device__ char * _strncpy_(char *dst, const char *src,register size_t n)
{
	if (n != 0) {
		register char *d = dst;
		register const char *s = src;

		do {
			if ((*d++ = *s++) == 0) {
				/* NUL pad the remaining n-1 bytes */
				while (--n != 0)
					*d++ = 0;
				break;
			}
		} while (--n != 0);
	}
	return (dst);
}



//Check each char inside this read, only "A/C/T/G" allowed in the fasta file
__device__ int PrepareSequence(char *read) 
{
	int p;
	
	int return_value = 1;
	
	for (p = 0; p < READ_LENGTH; p++ )
	{ 
		read[p] = _toupper_(read[p]);
		if (!(read[p] == 'A' ||
					read[p] == 'C' ||
					read[p] == 'T' || 
					read[p] == 'G'))
		{
			return_value = 0;
			break;
		}
	}
	return return_value;
}


//Check whether bloom filter contains "string key"
__device__ bool contains(char *key, unsigned int table_size)
{	
	
	unsigned int hash, bit, index,len;
	unsigned char bloom;
	
	unsigned int i;	
	unsigned int b    = 378551;
	unsigned int a    = 63689;
	
	len = TUPLE_SIZE;
	char str[TUPLE_SIZE+1];
	
	_strncpy_(str, key,TUPLE_SIZE);
	str[TUPLE_SIZE]=0;
	
	
    //_RSHash_	
	hash=0;i=0;
    for(i = 0; i < len; i++)
    {
      hash = hash * a + (str[i]);
      a    = a * b;
	}
	
	hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;    
    bloom = tex1Dfetch( tex, index);    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    
    //_JSHash_
     hash = 1315423911;
    i=0;
    for(i = 0; i < len; i++)
    {
      hash ^= ((hash << 5) + (str[i]) + (hash >> 2));
    }
    
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    
    //_PJWHash_
    unsigned int ThreeQuarters = (unsigned int)(((unsigned int)(sizeof(unsigned int) * 8)  * 3) / 4);
    unsigned int HighBits = (unsigned int)(0xFFFFFFFF) << (sizeof(unsigned int) * 7);
    hash= 0;
    a= 0;
    i= 0;
	
	for(i = 0; i < len; i++)
    {
      hash = (hash << sizeof(unsigned int)) + (str[i]);

      if((a = hash & HighBits)  != 0)
      {
         hash = (( hash ^ (a >> ThreeQuarters)) & (~HighBits));
      }
    }   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_ELFHash_
    hash=0;i=0;a=0;
    for(i = 0; i < len; i++)
   {
      hash = (hash << 4) + (str[i]);
      if((a = hash & 0xF0000000L) != 0)
      {
         hash ^= (a >> 24);
      }
      hash &= ~a;
   }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);  ;  
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_BKDRHash_
    hash=0;i=0;a=131;
    for(i = 0; i < len; i++)
    {
      hash = (hash * a) + (str[i]);
    }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_SDBMHash_
    hash=0;i=0;
    
    for(i = 0; i < len; i++)
   {
      hash = (str[i]) + (hash << 6) + (hash << 16) - hash;
   }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    
    //_DJBHash_
    hash = 5381;i=0;
    for(i = 0; i < len; i++)
   {
      hash = ((hash << 5) + hash) + (str[i]);
   }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    
    
    return true;
}

 __device__ bool contains2(char *key, unsigned int table_size)
{	
	
	unsigned int hash, bit, index,len;
	unsigned char bloom;
	
	unsigned int i;	
	
	
	len = TUPLE_SIZE;
	char str[TUPLE_SIZE+1];
	
	_strncpy_(str, key,TUPLE_SIZE);
	str[TUPLE_SIZE]=0;
 	
	//_DEKHash_
     hash = len;i=0;
    for(i = 0; i < len; i++)
    {
		hash = ((hash << 5) ^ (hash >> 27)) ^ (str[i]);
    }
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    /*
    //_BPHash_
    hash=0;i=0;
    for(i = 0; i < len; i++)
   {
      hash = hash << 7 ^ (str[i]);
   }
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_FNVHash_
   a = 0x811C9DC5;
   hash= 0;
   i= 0;

   for(i = 0; i < len; i++)
   {
      hash *= a;
      hash ^= (str[i]);
   }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_APHash_
    hash = 0xAAAAAAAA;i=0;
    for(i = 0; i < len; i++)
   {
      hash ^= ((i & 1) == 0) ? (  (hash <<  7) ^ (str[i]) * (hash >> 3)) :
                               (~((hash << 11) + (str[i]) ^ (hash >> 5)));
   }
   
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
  */
    
    return true;
 }
 
 __device__ bool contains3(char *key, unsigned int table_size)
{	
	
	unsigned int hash, bit, index,len;
	unsigned char bloom;
	
	unsigned int i;	
		
	len = TUPLE_SIZE;
	char str[TUPLE_SIZE+1];
	
	_strncpy_(str, key,TUPLE_SIZE);
	str[TUPLE_SIZE]=0;	
    
    //_krHash_
    hash = 0;
   for(i = 0; i < len; i++)
   {
        hash += str[i];
    }    
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);    
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_ocaml_hash_
     hash=0;i=0;
    for (i=0; i<len; i++) {
        hash = hash*19 + str[i];
    }
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);   
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_sml_hash_
    hash=0;i=0;
    for (i=0; i<len; i++) 
    {
        hash = 33*hash + 720 + str[i];
    }
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);   
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    //_stl_hash_
    _strncpy_(str, key,TUPLE_SIZE);
    hash=0;i=0;
    for (i=0; i<len; i++) 
    {
        hash = 5*hash + str[i];
    }
    hash = hash % (table_size * _char_size_);
    bit  = hash % _char_size_;
    index = hash / _char_size_ ;
    bloom = tex1Dfetch( tex, index);   
    
    if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
    {
		return false;
    }
    
    return true;
 }
 
//search tuple from bloon filter
__device__ int lstspct_FindTuple(char *tuple, int numTuples) 
{
		
		//check whether in the bloom filter
		//if(contains(tuple,numTuples * 4)&&contains2(tuple,numTuples * 4)&&contains3(tuple,numTuples * 4))
		if(contains(tuple,numTuples * BLOOM_SIZE)&&contains2(tuple,numTuples * BLOOM_SIZE))
			return 1;
		else
			return -1;
}

__device__ int d_strTpl_Valid(char *st) 
{
		int i;
		int return_value = 1;
				
		if (st == NULL)
			return_value = 0;
		else
		{
			for (i = 0; i < TUPLE_SIZE; i++) {
				if (numeric_nuc_index[st[i]] >= 4)
				{
					return_value = 0;
					break;
				}
			}			
		}
		
		return return_value;
}

//check whether the read is solid or not, by examine each tuple in this read, whether can be found or not in 
//all the string tuple list
__device__ int CheckSolid(char *seq, int tupleSize, int numTuples){
	int p;
	char tuple[TUPLE_SIZE+1];
	
	int return_value = 1;
		
	for (p = 0; p < READ_LENGTH - tupleSize +1; p++ ){
		_strncpy_(tuple, (char*) &seq[p],tupleSize);
		tuple[tupleSize] = 0;
		
		if (lstspct_FindTuple(tuple,numTuples) == -1) {
			return_value = 0;
			break;
		}
	}
	
	return return_value;
}



__device__ int SolidSubsequence(char *seq, int tupleSize, int &seqStart, int &seqEnd, int numTuples) {
	int i;
	int solidSubsequence = 1;
	
	//char tempTuple[TUPLE_SIZE+1];
	char *tempTuple;
	
	for (i = seqStart; i < seqEnd - tupleSize + 1; i++) 
	{				
		//_strncpy_(tempTuple , (char*) &seq[i],tupleSize);
		//tempTuple[tupleSize] = 0;
		
		tempTuple = &seq[i];
		if (lstspct_FindTuple(tempTuple,numTuples) == -1) {
			solidSubsequence = 0;
			break;
		}
	}
	

	return solidSubsequence;
}	


__device__ int TrimSequence(char *seq, int tupleSize, int &seqStart, int &seqEnd, int numTuples,int maxTrim) 
{	
	int i;
	seqStart = 0;
	
	int flag = 1;

	//char tempTuple[TUPLE_SIZE+1];
	char *tempTuple;
	
	//get length of this read
	int len = seq[READ_LENGTH + 1];
		
	for (i = 0; i < len - tupleSize + 1; i++ )
	{ 
		//_strncpy_(tempTuple , &seq[i],tupleSize);
		//tempTuple[tupleSize] = 0;
		
		tempTuple = &seq[i];
		
		if (lstspct_FindTuple(tempTuple,numTuples) != -1) {
			break;
		}
		
		// Not solid yet, advance
		seqStart++;
	}
	
	seqEnd = len;
	for (i = seqStart + 1; i < len - tupleSize + 1; i++ ) 
	{	
		//_strncpy_(tempTuple , &seq[i],tupleSize);
		//tempTuple[tupleSize] = 0;
		
		tempTuple = &seq[i];
		
		if (lstspct_FindTuple(tempTuple, numTuples) == -1) {
			break;
		}
	}
	
	if (i == len - tupleSize) 
		// The sequence is not trimmed.
		seqEnd = len - 1;
	else 
		// The sequence is trimmed. Trim end is the index of the first
		// 'bad' nucleotide. Since seqStart is the index of the first
		// 'good' nucleotide, seqEnd - seqStart is the length of the
		// untrimmed seq.  In other words, it's half open 0 based
		// indexing.
		seqEnd = i + tupleSize-1;
		
	if (seqStart > maxTrim)
	//		return 0;
		flag = 0;
	else if (len - seqEnd > maxTrim)
		//return 0;
		flag = 0;
	else if(SolidSubsequence(seq, tupleSize, seqStart, seqEnd,numTuples) == 0)
	//		return 0;
		flag = 0;
	else
	{		
		int newLength = seqEnd - seqStart + 1;
		
		for (int s = 0; s < newLength; s++ ) {
			seq[s] = seq[s + seqStart];
		}
		//seq.length = newLength -1;
		len = newLength -1;
	}
	
	//save the new length
	//_strncpy_(&seq[READ_LENGTH + 1], itoa1(len),4);
	//itoa1(len,&seq[READ_LENGTH + 1]);
	seq[READ_LENGTH + 1] = len;
	
	return flag;

}


////////////////////////////////////////////////////////////////////////////////
//! Fix two errors, step 1 kernel function
//! @param d_reads_arr  input data in global memory
//! @param d_param  input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void fix_errors1(char *d_reads_arr,Param *d_param) 
{
	short numSearch=1;
	nextNuc['G'] = 'A';	nextNuc['A'] = 'C';	nextNuc['C'] = 'T';	nextNuc['T'] = 'G';
	
	int c_tid = blockIdx.x * blockDim.x + threadIdx.x;		
	int round = 0;
	int total_thread = BLOCK * THREAD;
	int discardSeq=0;	
	int trimStart=0, trimEnd=0;
	
	int chunk_bound = (total_thread < MAX_READS_BOUND ? total_thread:MAX_READS_BOUND);	
	round = d_param->NUM_OF_READS/chunk_bound + (d_param->NUM_OF_READS%chunk_bound == 0 ? 0:1);
	
	int maxPos[READ_LENGTH * 4],maxMod[READ_LENGTH * 4];
	unsigned char votes[READ_LENGTH][4],mutNuc, mutNuc2, prev, cur;
	int solid[READ_LENGTH];
	int s,i,j,m,n,startPos, fixPos=-1,numFixed = 0,numChanges=0;	
	short return_value = 0,flag = 0,flag1=1;
	
	// Cast votes for mutations
	int p,vp,mut;
	short numAboveThreshold = 0,newLength,len;
	short maxVotes = 0,allGood  = 1;
	int numTies = -1,pindex = 0, mod, pos,current_read_idx;
	
	char *tempTuple, *read;
		
	for(i=0;i<round;i++)
	{
		flag = 0;flag1=1;numFixed = 0;	numChanges=0;	return_value = 0;discardSeq = 0;
		
		current_read_idx = c_tid + chunk_bound * i;
		
		//check if run out of reads		
		current_read_idx = (current_read_idx > d_param->NUM_OF_READS ? 0:current_read_idx);		
		
		//take 1 read per thread	
		read = &d_reads_arr[current_read_idx*(READ_LENGTH + 2)];
		
		//get length of this read
		len = read[READ_LENGTH + 1];		
		
		if (!PrepareSequence(read)) {
			discardSeq = 1;
		}
		else 
		{			
			numFixed = 0; fixPos = -1;
			do{				
				if(flag)
					break;
				else{
					if (fixPos > 0)
						startPos = fixPos;
					else 
						startPos = 0;
														
					for (m = 0; m < READ_LENGTH; m++) {
						for (int n = 0; n < 4; n++) 
							//votes[threadIdx.x][m][n] = 0;
							votes[m][n] = 0;
					}					
							
					for(m=0;m<READ_LENGTH;m++)
						solid[m] = 0;
						
					for (p = startPos; p < len - d_param->tupleSize + 1; p++ ){							
						tempTuple = &read[p];
						if (d_strTpl_Valid(tempTuple)){
							if (lstspct_FindTuple(tempTuple, d_param->numTuples) != -1) 
								solid[p] = 1;							
							else{								
								for (vp = 0; vp < d_param->tupleSize; vp++){										
									mutNuc = nextNuc[read[p + vp]];									
									read[p + vp] = mutNuc;
											
									for (mut = 0; mut < 3; mut++ ){											
										tempTuple = &read[p];
											
										if (lstspct_FindTuple(tempTuple, d_param->numTuples) != -1)										
											votes[vp + p][unmasked_nuc_index[mutNuc]]++;																										
											
										mutNuc = nextNuc[mutNuc];						
										read[p + vp] = mutNuc;											
									}
								}
							}
						}
					}
		
					////////////////vote completed//////////////////////						
					++numFixed;	
								
					//////////////////////fix sequence based on voting in previous step//////////////
					fixPos = 0;numAboveThreshold = 0;maxVotes = 0;allGood  = 1;
	
					for (p = 0; p < len - d_param->tupleSize + 1; p++ )	{
						if (solid[p] == 0) {
							allGood = 0;break;
						}
					}
						
					if (allGood)
						// no need to fix this sequence						
						return_value =  1;					
					else
					{					
						for (p = 0; p < len; p++){ 
							for (m = 0; m < 4; m++){								
								if (votes[p][m] > d_param->minVotes)
									numAboveThreshold++;												
								
								if (votes[p][m] >= maxVotes)																
									maxVotes = votes[p][m];								
							}
						}
								
						pindex = 0;numTies = -1;
						
						// Make sure there aren't multiple possible fixes
						for (p = 0; p < len; p++){ 
							for (m = 0; m < 4; m++){
								if (votes[p][m] == maxVotes){
									numTies++;
									maxPos[pindex] = p;
									maxMod[pindex] = m;
									pindex++;
								}
							}
						}
													
						if (numAboveThreshold > 0 ){							
							if (numTies < numSearch || (pindex > 1 && maxPos[0] != maxPos[1])){								
								// Found at least one change to the sequence										
								for (s = 0; s < numSearch && s < pindex; s++) {
									mod = maxMod[s];
									pos = maxPos[s];
									fixPos = pos;
									
									if (mod < 4){
										prev = read[pos];
										cur = nuc_char[mod];
										read[pos] = cur;
									}									
								}
								if( CheckSolid(read,d_param->tupleSize,d_param->numTuples))
									return_value = 1;
								else{
									//reset
									return_value = 0;
									//read[pos] = prev;
								}							
							} 
							else 
							{									
								return_value = 0;
							}
						}
						else 
						{
							return_value = 0;
						}
					}
					
					
					//check fix sequence return
					if( return_value)
					{
						flag = 1;
						numChanges = numFixed;
						break;
					}					
				}			
			} while (fixPos > 0);
	
			/////////////////////////end of solidify////////

			if (numChanges != 0){
				if (numChanges > d_param->maxMods) 
					discardSeq = 1;
				else
					discardSeq = 0;
			}
			else{
				if( d_param->numSearch == 2){
					//removed trim in fix error1
					discardSeq = 1;
				}
				else
				{
					// Find the locations of the first solid positions.
					if (d_param->doTrim)
					{					
						if(TrimSequence(read, d_param->tupleSize,trimStart, trimEnd, d_param->numTuples,d_param->maxTrim)){
							// If there is space for one solid tuple (trimStart < trimEnd - ts+1)
							// and the subsequence between the trimmed ends is ok, print the
							// trimmed coordinates.						
							discardSeq = 0;
						}
						else
							discardSeq = 1;	
					}
					else				
						discardSeq = 1;				
				}
			}
		}
		
		if (discardSeq) {
			read[READ_LENGTH] = 'D'; //F fixed, D: not fixed, discard				
		}
		else {
			read[READ_LENGTH] = 'F'; //F fixed, D: not fixed, discard
		}		
		
		__syncthreads();	
	}	
}

////////////////////////////////////////////////////////////////////////////////
//! Fix two errors step 2 kernel function
//! @param d_reads_arr  input data in global memory
//! @param d_param  input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void fix_errors2(char *d_reads_arr,Param *d_param, int numReads) 
{
	short numSearch = 2;
	
	nextNuc['G'] = 'A';	nextNuc['A'] = 'C';	nextNuc['C'] = 'T';	nextNuc['T'] = 'G';
	
	int c_tid = blockIdx.x * blockDim.x + threadIdx.x;		
	int round = 0;
	int total_thread = BLOCK * THREAD;
	int discardSeq=0;	
	int trimStart=0, trimEnd=0;
	
	int chunk_bound = (total_thread < MAX_READS_BOUND ? total_thread:MAX_READS_BOUND);	
	round = numReads/chunk_bound + (numReads%chunk_bound == 0 ? 0:1);
	
	int maxPos[READ_LENGTH * 4];
	int maxMod[READ_LENGTH * 4];
	
	unsigned char votes[READ_LENGTH][4],mutNuc, mutNuc2, prev, cur;
	
	int solid[READ_LENGTH];
	//__shared__ unsigned char solid[READ_LENGTH];
	
	int s,i,j,m,n;
	int startPos, fixPos=-1;
	int numFixed = 0,numChanges=0;	
	short return_value = 0,flag = 0,flag1=1;
	
	// Cast votes for mutations
	int p,vp,mut;
	short numAboveThreshold = 0;
	short maxVotes = 0,allGood  = 1;
	int numTies = -1;
	int pindex = 0;
	int mod, pos;	
	short newLength,len;	
	int current_read_idx;
	
	char *tempTuple, *read;
	
	/*
	Since GPU cannot process all reads at the same time (limited block NO.), the reads are divided 
	into several rounds to process.		
	*/	
	for(i=0;i<round;i++)
	{
		flag = 0;	flag1=1;numFixed = 0;numChanges=0;return_value = 0;
		
		current_read_idx = c_tid + chunk_bound * i;
		
		//check if run out of reads		
		current_read_idx = (current_read_idx > numReads ? 0:current_read_idx);	
		
		//take 1 read per thread
		read = &d_reads_arr[current_read_idx*(READ_LENGTH + 2)];

		//get length of this read
		len = read[READ_LENGTH + 1];
		
		discardSeq = 0;
		
		if (!PrepareSequence(read)) 
			discardSeq = 1;		
		else 
		{
			numFixed = 0; fixPos = -1;
			do {				
				if(flag)
					break;				
				else{
					if (fixPos > 0)
						startPos = fixPos;
					else 
						startPos = 0;
									
						for (m = 0; m < READ_LENGTH; m++) {
							for (int n = 0; n < 4; n++) 
								votes[m][n] = 0;
						}
					
							
						for(m=0;m<READ_LENGTH;m++)
							solid[m] = 0;
						
						for (p = startPos; p < len - d_param->tupleSize + 1; p++ ) {
													
							tempTuple = &read[p];
							if (d_strTpl_Valid(tempTuple)) {
								if (lstspct_FindTuple(tempTuple, d_param->numTuples) != -1) 
									solid[p] = 1;								
								else{									
									for (vp = 0; vp < d_param->tupleSize-1; vp++) 
									{
										
										mutNuc = nextNuc[read[p + vp]];										
										read[p + vp] = mutNuc;
											
										for (mut = 0; mut < 3; mut++ ) 
										{					
										
											tempTuple = &read[p];
											
											if (lstspct_FindTuple(tempTuple, d_param->numTuples) != -1) 
											{
												votes[vp + p][unmasked_nuc_index[mutNuc]]++;
											}
											
											//delta = 2
											for(m=vp+1;m<d_param->tupleSize;m++)
											{
												mutNuc2 = nextNuc[read[p + m]];
												read[p + m] = mutNuc2;	
												
												for(n=0;n<3;n++)
												{
													tempTuple = &read[p];
													if (lstspct_FindTuple(tempTuple, d_param->numTuples) != -1) 
													{														
														votes[vp + p][unmasked_nuc_index[mutNuc]]++;//history
														votes[m + p][unmasked_nuc_index[mutNuc2]]++;								
													}
													
													mutNuc2 = nextNuc[mutNuc2];
													read[p + m] = mutNuc2;																																					
												}
											}											
											
											mutNuc = nextNuc[mutNuc];						
											read[p + vp] = mutNuc;
											
										}
									}
								}
							}
						}
		
						
						++numFixed;
						
						//fix sequence based on voting in previous step
						fixPos = 0;numAboveThreshold = 0;maxVotes = 0;allGood  = 1;
	
						for (p = 0; p < len - d_param->tupleSize + 1; p++ ) {
							if (solid[p] == 0) {
								allGood = 0;
								break;
							}
						}
						
						
						if (allGood) 						
							// no need to fix this sequence						
							return_value =  1;						
						else{
							for (p = 0; p < len; p++){ 
								for (m = 0; m < 4; m++) {									
									if (votes[p][m] > d_param->minVotes)									 
										numAboveThreshold++;
								
									if (votes[p][m] >= maxVotes)										
										maxVotes = votes[p][m];									
								}
							}
								
							pindex = 0;numTies = -1;
							
							for (p = 0; p < len; p++){ 
								for (m = 0; m < 4; m++)	{									
									if (votes[p][m] == maxVotes) {
										numTies++;
										
										maxPos[pindex] = p;
										maxMod[pindex] = m;
										pindex++;
									}
								}
							}
							
							//second 
							votes[p][m] = 0;
							maxVotes = 0;
							for (p = 0; p < len ; p++){ 
								for (m = 0; m < 4; m++) {
																			
									if (votes[p][m] >= maxVotes)										
										maxVotes = votes[p][m];									
								}
							}
							for (p = 0; p < len; p++){ 
								for (m = 0; m < 4; m++)	{									
									if (votes[p][m] == maxVotes) {										
										maxPos[pindex] = p;
										maxMod[pindex] = m;	
										//pindex++;									
									}
								}
							}
							
							__syncthreads();					
							if (numAboveThreshold > 0 ) 
							{							
								//if (numTies < numSearch || (pindex > 1 && maxPos[0] != maxPos[1])){								
									// Found at least one change to the sequence										
									for (s = 0; s < 2; s++) {
										mod = maxMod[s];
										pos = maxPos[s];
										fixPos = pos;
									
										if (mod < 4){
											prev = read[pos];
											cur = nuc_char[mod];
											read[pos] = nuc_char[mod];											
										}
									}
														
									return_value = CheckSolid(read,d_param->tupleSize,d_param->numTuples);	
								//} 
								//else {									
									return_value = 0;
								//}
							}
							else {
								return_value = 0;
							}
							__syncthreads();
						}
					
						//check fix sequence return
						if( return_value){
							flag = 1;
							numChanges = numFixed;
							break;
						}

				}//if flag		
			} while (fixPos > 0);
	
			/////////////////////////end of solidify////////

			if (numChanges != 0) {
				if (numChanges > d_param->maxMods){	
					discardSeq = 1;
					//_strncpy_(read , original, READ_LENGTH + 2);
				}
				else
				{
					discardSeq = 0;
				}
			}			
			else
			{	
				
				// Find the locations of the first solid positions.
				if (d_param->doTrim){					
					if(TrimSequence(read, d_param->tupleSize,trimStart, trimEnd, d_param->numTuples,d_param->maxTrim)){
						// If there is space for one solid tuple (trimStart < trimEnd - ts+1)
						// and the subsequence between the trimmed ends is ok, print the
						// trimmed coordinates.
						
						discardSeq = 0;
					}
					else
						discardSeq = 1;									
					
				}
				else 
				{				
					discardSeq = 1;
				}
				
			}
		}		
		
		if (discardSeq) {
			read[READ_LENGTH] = 'D'; //last char for indicator
		}
		else {
			read[READ_LENGTH] = 'F'; //F fixed, D: not fixed, discard
		}
		
		__syncthreads();	
	}
	
}

#endif // #ifndef _FIXERRORSVOTING_KERNEL_H_
                                                                                                                                                                       hash_function.cpp                                                                                   0000644 0000765 0000765 00000011302 11167253302 014563  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               #include "hash_function.h"

unsigned int RSHash(char* str, unsigned int len)
{
   unsigned int b    = 378551;
   unsigned int a    = 63689;
   unsigned int hash = 0;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = hash * a + (*str);
      a    = a * b;
   }

   return hash;
}
/* End Of RS Hash Function */


unsigned int JSHash(char* str, unsigned int len)
{
   unsigned int hash = 1315423911;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash ^= ((hash << 5) + (*str) + (hash >> 2));
   }

   return hash;
}
/* End Of JS Hash Function */


unsigned int PJWHash(char* str, unsigned int len)
{
   unsigned int BitsInUnsignedInt = (unsigned int)(sizeof(unsigned int) * 8);
   unsigned int ThreeQuarters     = (unsigned int)((BitsInUnsignedInt  * 3) / 4);
   unsigned int OneEighth         = (unsigned int)(BitsInUnsignedInt / 8);
   unsigned int HighBits          = (unsigned int)(0xFFFFFFFF) << (BitsInUnsignedInt - OneEighth);
   unsigned int hash              = 0;
   unsigned int test              = 0;
   unsigned int i                 = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = (hash << OneEighth) + (*str);

      if((test = hash & HighBits)  != 0)
      {
         hash = (( hash ^ (test >> ThreeQuarters)) & (~HighBits));
      }
   }

   return hash;
}
/* End Of  P. J. Weinberger Hash Function */


unsigned int ELFHash(char* str, unsigned int len)
{
   unsigned int hash = 0;
   unsigned int x    = 0;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = (hash << 4) + (*str);
      if((x = hash & 0xF0000000L) != 0)
      {
         hash ^= (x >> 24);
      }
      hash &= ~x;
   }

   return hash;
}
/* End Of ELF Hash Function */


unsigned int BKDRHash(char* str, unsigned int len)
{
   unsigned int seed = 131; /* 31 131 1313 13131 131313 etc.. */
   unsigned int hash = 0;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = (hash * seed) + (*str);
   }

   return hash;
}
/* End Of BKDR Hash Function */


unsigned int SDBMHash(char* str, unsigned int len)
{
   unsigned int hash = 0;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = (*str) + (hash << 6) + (hash << 16) - hash;
   }

   return hash;
}
/* End Of SDBM Hash Function */


unsigned int DJBHash(char* str, unsigned int len)
{
   unsigned int hash = 5381;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = ((hash << 5) + hash) + (*str);
   }

   return hash;
}
/* End Of DJB Hash Function */


unsigned int DEKHash(char* str, unsigned int len)
{
   unsigned int hash = len;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash = ((hash << 5) ^ (hash >> 27)) ^ (*str);
   }
   return hash;
}
/* End Of DEK Hash Function */


unsigned int BPHash(char* str, unsigned int len)
{
   unsigned int hash = 0;
   unsigned int i    = 0;
   for(i = 0; i < len; str++, i++)
   {
      hash = hash << 7 ^ (*str);
   }

   return hash;
}
/* End Of BP Hash Function */


unsigned int FNVHash(char* str, unsigned int len)
{
   unsigned int fnv_prime = 0x811C9DC5;
   unsigned int hash      = 0;
   unsigned int i         = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash *= fnv_prime;
      hash ^= (*str);
   }

   return hash;
}
/* End Of FNV Hash Function */


unsigned int APHash(char* str, unsigned int len)
{
   unsigned int hash = 0xAAAAAAAA;
   unsigned int i    = 0;

   for(i = 0; i < len; str++, i++)
   {
      hash ^= ((i & 1) == 0) ? (  (hash <<  7) ^ (*str) * (hash >> 3)) :
                               (~((hash << 11) + (*str) ^ (hash >> 5)));
   }

   return hash;
}
/* End Of AP Hash Function */


unsigned int krHash(char *str, unsigned int len)
{
      unsigned int hash = 0;
      int c;
    
      while (c = *str++)
        hash += c;
                      
      return hash;
} 
/**/
unsigned int ocaml_hash(char *str, unsigned int len) 
    {
      unsigned int hash = 0;
      unsigned int i;
    
      for (i=0; i<len; i++) {
        hash = hash*19 + str[i];
      }
    
      return hash;
    }

unsigned int sml_hash(char *str, unsigned int len) {
      unsigned int hash = 0;
      unsigned int i;
    
      for (i=0; i<len; i++) 
      {
        hash = 33*hash + 720 + str[i];
      }
    
      return hash;
    }


 unsigned int stl_hash(char *str, unsigned int len) 
    {
      unsigned int hash = 0;
      unsigned int i;
    
      for (i=0; i<len; i++) 
      {
        hash = 5*hash + str[i];
      }
    
      return hash;
    }

                                                                                                                                                                                                                                                                                                                              hash_function.h                                                                                     0000644 0000765 0000765 00000002063 11167253302 014234  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               #ifndef INCLUDE_HASHFUNCTION_C_H
#define INCLUDE_HASHFUNCTION_C_H


#include <stdio.h>


//15
extern "C" unsigned int RSHash  (char* str, unsigned int len);
extern "C" unsigned int JSHash  (char* str, unsigned int len);
extern "C" unsigned int PJWHash (char* str, unsigned int len);
extern "C" unsigned int ELFHash (char* str, unsigned int len);
extern "C" unsigned int BKDRHash(char* str, unsigned int len);
extern "C" unsigned int SDBMHash(char* str, unsigned int len);
extern "C" unsigned int DJBHash (char* str, unsigned int len);
extern "C" unsigned int DEKHash (char* str, unsigned int len);
extern "C" unsigned int BPHash  (char* str, unsigned int len);
extern "C" unsigned int FNVHash (char* str, unsigned int len);
extern "C" unsigned int APHash  (char* str, unsigned int len);
extern "C" unsigned int krHash(char *str, unsigned int len);
extern "C" unsigned int ocaml_hash(char *str, unsigned int len) ;
extern "C" unsigned int sml_hash(char *str, unsigned int len);
extern "C" unsigned int stl_hash(char *str, unsigned int len);


#endif
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             main.cpp                                                                                            0000664 0000765 0000765 00000025644 11167261236 012704  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               /*
main cpp, pre-processing, call cuda .cu file
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include <math.h>

#include "cutil.h"

// includes, project
#include "FixErrorsVoting.h"
#include "hash_function.h"

#include "utils.h"

#include <time.h> //clock()
#include <vector>
#include <string>

//using namespace std;
using std::string;
using std::vector;

//#include <conio.h>
extern "C" void runTest(unsigned char *hash_table,Param *h_param,char *reads_arr,unsigned int table_size, 
						FILE *seqOut, 
						FILE *discardOut,
						int argc,
						char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
   clock_t start,end;//clock 
	
    int minMult=6;
	short doInsertion=0, doDeletion=0, doTrim=1, maxTrim = 20;
	short maxMods=999,numSearch = 1, minVotes=2;
	int NUM_READS=0, i,j,k,m;
	
	//Vector of strings, used to store tuples with multiplicity > minMult
	vector<string> tupleList;
	FILE *seqIn,*seqOut,*discardOut;
	int numTuples=0,readLen=35,numPastThresh=0;	
	unsigned int hash, hash_vector[NUM_HASH];
	//fasta file
	char readsFile[50];
	//tuple size
	short tupleSize;
	//fixed file name
	char outputFileName[50];
	char discardFileName[50];

	//check parameters	
   int argi;
	if (argc < 4) {
		PrintUsage();
		exit(1);
	}
	argi = 1;
	
	//process input parameters
	while (argi < argc) {
		if (strcmp(argv[argi], "-f") == 0 ){
			++argi;
			strcpy(readsFile,argv[argi]);
		} else if (strcmp(argv[argi], "-t") == 0 ) {
	      ++argi;
	      tupleSize = atoi(argv[argi]);
	    } else if (strcmp(argv[argi], "-o") == 0 ) {
	      ++argi;
	      strcpy(outputFileName,argv[argi]);
	    }else if (strcmp(argv[argi], "-d") == 0 ) {
	      ++argi;
	      strcpy(discardFileName , argv[argi]);
	    }else if (strcmp(argv[argi], "-r") == 0) {
			readLen = atoi(argv[++argi]);
		}else if (strcmp(argv[argi], "-minMult") == 0 ) {
	      ++argi;
	      minMult = atoi(argv[argi]);
	    }else if (strcmp(argv[argi], "-minVotes") == 0 ) {
	      ++argi;
	      minVotes = atoi(argv[argi]);
	    }else if (strcmp(argv[argi], "-search") == 0) {
			numSearch = atoi(argv[++argi]);
		}
		else if (strcmp(argv[argi], "-deletions") == 0) {
			doDeletion = 1;
		}
		else if (strcmp(argv[argi], "-insertions") == 0) {
			doInsertion = 1;
		}
		else if (strcmp(argv[argi], "-trim") == 0) {
			doTrim = 1;
		}		
		else if (strcmp(argv[argi], "-maxTrim") == 0) {
			doTrim = 1;
			maxTrim = atoi(argv[++argi]);
		}		
		else if (strcmp(argv[argi], "-maxMods") == 0) {
			maxMods = atoi(argv[++argi]);
		}else {
			PrintUsage();
			printf( "bad option: %c \n" , argv[argi] );
			exit(0);
		}
		argi++;
	}
	
	printf("Running.... in main....readLen: %d, numSearch: %d\n", readLen, numSearch);

	seqOut = fopen(outputFileName,"w");

	seqIn = fopen(readsFile,"r");
	if(!seqIn)
	{
		printf("Please check your fasta file name, FILE not exist! \n\n");
		exit(0);
	}
	
	if (discardFileName != "") 
		discardOut = fopen(discardFileName,"w");	
	
	//read and its reverse 
	char *Sequence = new char[readLen+1];
	char *Rc_sequence = new char[readLen+1];		
	char *tempTuple = new char[readLen+1];	

	/*
	**Read from the fasta file and store the comment and read in a DNASequence vector. 
	*/
	printf( "Reading sequence from fasta file...");	
	
	vector<DNASequence> ReadsList;
//	DNASequence *read = (DNASequence *)malloc(sizeof(DNASequence));
	DNASequence read;
	
	while (GetSeq(seqIn, &read)) 
	{
		ReadsList.push_back(read);
		NUM_READS ++;

		//printf("Reads: %d\n", NUM_READS);
	}
	NUM_READS --;
	printf( "Read seq done....NUM_READS: %d \n",NUM_READS);
	

	/*Preprocessing. Counting tuples with mult > minMult.
	**Counting each read and its reverse from the fasta file
	**Using bloom filter to do the counting. Total minMult number of bloom filters needed
	*/	
	start = clock();
	//hash_function hash_function_list[]     = {&RSHash,&JSHash,&PJWHash,&ELFHash,&BKDRHash,&SDBMHash,&DJBHash,&DEKHash,&BPHash,&FNVHash,&APHash,&krHash,&ocaml_hash,&sml_hash,&stl_hash};
	hash_function hash_function_list[]     = {&RSHash,&JSHash,&PJWHash,&ELFHash,&BKDRHash,&SDBMHash,&DJBHash,&DEKHash};
		
	/*
	**Bloom filter size. For large dataset, will be smaller.
	**6 bloom filters in the same hash position.
	**k=8, 4:m/n=32,5.73e-06; k=4,m/n=32,0.000191;k=4,m/n=64,1.35e-5
	*/	
	short BLOOM1_BIT;

	if(NUM_READS < 6000000) 
		BLOOM1_BIT = 4; // 
	else
		BLOOM1_BIT = 1;

	unsigned int ext_filter_size = NUM_READS * (readLen - tupleSize + 1) * BLOOM1_BIT * minMult;
	unsigned int filter_size = NUM_READS * (readLen - tupleSize + 1) * BLOOM1_BIT;
	
	unsigned char *bloom1 = (unsigned char *)malloc(sizeof(unsigned char)*ext_filter_size);
	if(!bloom1){
		printf("Cannot allocate bloom1 memory on CPU\n");
		exit(-1);
	}
	else{
		printf( "Alloc bloom1 memory done..., %d MB allocated\n",sizeof(unsigned char)*ext_filter_size/(1024*1024));
	}

	//reset bloom filter
    for(i = 0; i < ext_filter_size; ++i){
		bloom1[i] = 0;
	}
		
	
	for(i = 0; i< NUM_READS; i ++){		
		strncpy(Sequence,ReadsList[i].seq,READ_LENGTH);
		Sequence[readLen] = 0;
		
		//printf("Processing read %d \n",i);
		for(j=0; j< readLen - tupleSize + 1; j++){
			//each tuple
			strncpy(tempTuple, &Sequence[j],TUPLE_SIZE);
			tempTuple[tupleSize] = '\0';
			
			//get hash values for this tuple
			for(m= 0; m < NUM_HASH; m++) 
				hash_vector[m] = hash_function_list[m](tempTuple,tupleSize) % (filter_size * char_size);			
			
			//each hash position holds 6 bits
			if(bfQuery(hash_vector,bloom1,5,minMult)){
				//do nothing				
			}else if(bfQuery(hash_vector,bloom1,4,minMult))
			{
				tupleList.push_back(tempTuple);
				numPastThresh ++;
				bfInsert(hash_vector,bloom1,5,minMult);
			}else if(bfQuery(hash_vector,bloom1,3,minMult))
				bfInsert(hash_vector,bloom1,4,minMult);
			else if(bfQuery(hash_vector,bloom1,2,minMult))
				bfInsert(hash_vector,bloom1,3,minMult);
			else if(bfQuery(hash_vector,bloom1,1,minMult))
				bfInsert(hash_vector,bloom1,2,minMult);
			else if(bfQuery(hash_vector,bloom1,0,minMult))
				bfInsert(hash_vector,bloom1,1,minMult);
			else 
				bfInsert(hash_vector,bloom1,0,minMult);			
		}
		
		//reverse count
		MakeRC(Sequence, readLen,Rc_sequence);
		Rc_sequence[readLen] = 0;	
		
		for(j=0; j< readLen - tupleSize + 1; j++){
			//each tuple
			strncpy(tempTuple, &Rc_sequence[j],tupleSize);
			tempTuple[tupleSize] = '\0';			
			
			//get hash values for this tuple
			for(m= 0; m < NUM_HASH; m++){
				hash_vector[m] = hash_function_list[m](tempTuple,TUPLE_SIZE) % (filter_size * char_size);
			}
			
			//each hash position holds 6 bits
			if(bfQuery(hash_vector,bloom1,5,minMult)){
				//do nothing
			}else if(bfQuery(hash_vector,bloom1,4,minMult)){
				tupleList.push_back(tempTuple);
				numPastThresh ++;
				bfInsert(hash_vector,bloom1,5,minMult);
			}else if(bfQuery(hash_vector,bloom1,3,minMult))
				bfInsert(hash_vector,bloom1,4,minMult);
			else if(bfQuery(hash_vector,bloom1,2,minMult))
				bfInsert(hash_vector,bloom1,3,minMult);
			else if(bfQuery(hash_vector,bloom1,1,minMult))
				bfInsert(hash_vector,bloom1,2,minMult);
			else if(bfQuery(hash_vector,bloom1,0,minMult))
				bfInsert(hash_vector,bloom1,1,minMult);
			else 
				bfInsert(hash_vector,bloom1,0,minMult);
										
		}
	}	
	
	printf("Total %d past threshhold (minMult = %d) \n", numPastThresh,minMult);
	
	free(bloom1);	

	//Building Bloom Filter	for GPU cuda
	unsigned int table_size = numPastThresh * BLOOM_SIZE;//m bits, n keys, m/n=32, 1 char = 8 bits, 1 char * 4 = 32 bits, increase to 64
	
    unsigned char *hash_table = (unsigned char *)malloc(sizeof(unsigned char)*table_size);
    for(unsigned int i = 0; i < table_size; ++i) 
		hash_table[i] = 0;
		
	for(i=0;i<numPastThresh;i++){
		//insert
		for(j = 0; j < NUM_HASH; j++) {
			 strcpy(tempTuple,tupleList[i].c_str());
			 
			 hash = hash_function_list[j](tempTuple,TUPLE_SIZE) % (table_size * char_size);
			 hash_table[hash / char_size] |= bit_mask[hash % char_size];
		  }
    }
	
	//free memory, vector string
	delete []tempTuple;
	delete []Sequence;
	delete []Rc_sequence;
	tupleList.clear();
	
	end = clock();
	
	printf("The run time for buliding bloom filter is: %f secs.\n", (double)(end-start)/CLOCKS_PER_SEC);
	

	/* Prepare host side reads array.
	** Each read consist of DNA sequence + 1 char (flag) + 1 char (read length)
	*/
	int arr_size = (readLen + 2)*NUM_READS; //1 byte for fixed flag, others for read char, 1 byte for read length
	printf( "arr_size %d \n",arr_size);
	
	char *reads_arr = (char *)malloc(sizeof(char)*arr_size);	
	if(reads_arr== NULL){
		printf("Malloc reads array memory failed \n");
		exit(1);
	}
	
	printf("Malloc reads array memory finished \n"); 

	//prepare reads array in CPU
	for(i = 0; i< NUM_READS; i ++){
		//init reads array
		strncpy(&reads_arr[i*(readLen + 2)],ReadsList[i].seq , (readLen+1));
		//init the 'fixed' flag as all discarded 'D'
		reads_arr[i*(readLen + 2) + readLen] = 'D';
		//init read length, last 1 BYTE		
		reads_arr[i*(readLen + 2) + readLen + 1] = ReadsList[i].length;		
	}	

	Param *h_param = (Param *)malloc(sizeof(Param));
	h_param->tupleSize = tupleSize;
	h_param->doTrim = doTrim;
	h_param->doDeletion = doDeletion;
	h_param->doInsertion = doInsertion;
	h_param->maxMods = maxMods;
	h_param->minVotes = minVotes;
	h_param->maxTrim = maxTrim;
	h_param->numTuples = numPastThresh;	
	h_param->numSearch = numSearch;
	h_param->NUM_OF_READS = NUM_READS;
	h_param->readLen = readLen;
	printf( "Alloc Param done....\n");	
	
	start=clock();//caculate GPU time
	
	/*********************************************************************************/
	/**************************** Call CUDA main function ****************************/
	//runTest(hash_table,h_param,h_iReadSeq,table_size);
	runTest(hash_table,h_param,reads_arr,table_size,seqOut,discardOut,argc,argv);
	
	/*********************************************************************************/

	end = clock();	
	printf("The run time for fixing error in GPU is: %f secs.\n", (double)(end-start)/CLOCKS_PER_SEC);

	printf("Write output to file....\n");

	//write to fixed file
	FILE *fout;	
	int len;

	for(int i=0;i<NUM_READS;i++){	
		if(reads_arr[i*(readLen + 2)+readLen]== 'D')
			fout = discardOut;
		else
			fout = seqOut;
				  
		fprintf(fout, ">%s\n", ReadsList[i].namestr);		
		len = reads_arr[i*(readLen + 2) + readLen + 1];

		for (j = 0; j < len; j++) 
		{
			fprintf(fout,"%c", reads_arr[i*(readLen + 2)+j]);
		}
		fprintf(fout,"\n");		
	}
	
	
	// cleanup memory
	printf("...releasing CPU memory.\n");	
	free(hash_table);
	free(reads_arr);
	ReadsList.clear();
	free(h_param);
	
	//free(read);		
	fclose(seqOut);
	fclose(discardOut);
}

                                                                                            Makefile                                                                                            0000644 0000765 0000765 00000004152 11167253302 012674  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               ################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= CUDA-EC
# CUDA source files (compiled with cudacc)
CUFILES		:= FixErrorsVoting.cu
# CUDA dependency files
CU_DEPS		:= \
	FixErrorsVoting_kernel.cu \

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	hash_function.cpp \
	utils.cpp \
	main.cpp \
	

################################################################################
# Rules and targets

include ../../common/common.mk
                                                                                                                                                                                                                                                                                                                                                                                                                      README.txt                                                                                          0000644 0000765 0000765 00000005566 11167243273 012752  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               CUDA-EC: A Parallel Fast error correction tool for high-throughput short-reads DNA Sequence.

CUDA-EC v.1.0
Contact: hxshi@ntu.edu.sg

---------
Overview:
---------
CUDA-EC is a C program for fixing errors for short-read hight-throughput data,
such as those genrated by the Illumina Genome Analyzer. 

----------
Reference:
----------
Haixiang Shi, Bertil Schmidt, Weiguo Liu, and Wolfgang M��ller-Wittig: "A Parallel Algorithm for Error Correction in High-Throughput Short-Read Data on CUDA-enabled Graphics Hardware", Manuscript submitted 

-------------------
System Requirments: 
-------------------
CUDA-EC has been tested on systems running Linux and CUDA version 1.1 with Nvidia GeForce GTX 280. 

-------------
Installation: 
-------------
Install CUDA toolkit v1.1 and CUDA SDK v1.1
Unpack CUDA-EC.tar into CUDA SDK directory $CUDA_SDK_DIR/NVIDIA_CUDA_SDK/projects
Compile CUDA-EC using make, generate executable file in $CUDA_SDK_DIR/bin/linux/release

------
Usage: 
------
./CUDA-EC -f {inputfilename} -t {tuplesize} -o {fixed-filename} -d {discarded-filename} -r {read_length}[-maxTrim {maximum_trim}] [-minVotes {minimum votes}] [-minMult {multiplicity}] [-search {num_error_to_fix]]


--------------------
REQUIRED PARAMETERS:   
-------------------- 
-f {inputfilename} 		--	Path and name of the input file containing the reads in FASTA format. 

In the current CUDA-EC version, all reads must be of equal length, with reads comments and reads each line, and contain only the letters {A,C,G,T}; e.g.
>Read:0
CACCTCAGAATTAACCCCACGCGGGCAGTCTGATA
>Read:1
TTCCACTGCTGCTCCCGCTTTGTCACCAGAAGAAA
>Read:2
GAAATTGAGACGAGACGCCAAAATAAAAAGAAAAA

     .....

-t {tuplesize}							--	length of tuple
-o {fixed-filename}	  			-- 	name of output fixed file
-d {discarded-filename}			--	name of output discarded file
-r {read_length}						--	length of the input reads

--------------------
OPTIONAL PARAMETERS: 
--------------------
-maxTrim: 								  --	Maximum number of trimmed character allowed (default 20)
-minVotes      							-- a threshold value for number of votes (default 2)
-minMult       						  -- multiplicity (default 6)
-search        							-- number of error to be fixed in each read (default 1)


--------
OUTPUT:
--------

CUDA-EC outputs two files in the same directiroy as the input file, one is the fixed file contains the fixed and trimmed reads, the another is the file contains the discarded reads. 

--------
Example:
--------
On the CUDA-EC homepage you will find a dataset with simulated read that was obtained by randomly sampling 1.1M reads of 35 bases from the 576,869bp S.cer5 (NC_00137) genome. A base error rate of 1% has been uniformly introduced in the read sequences.
Dowonlad the file and unpack it to the same directoy as CUDA-EC. CUDA-EC can now be called as follows:

./CUDA-EC  -f NC001137_e001.fasta -t 20 -o NC001137_e001.fasta.fixed -d NC001137_e001.fasta.discards -r 35 -search 1
 




                                                                                                                                          utils.cpp                                                                                           0000664 0000765 0000765 00000012621 11167253302 013102  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "common.h"

	

void MakeRC(char* seq, int len, char* &rev){
  int s;
  char n, c;
 
  for (s = 0; s < len; s++) {
    n = seq[len-s-1];
    c = comp_ascii[n];
    if (c != 0)
      rev[s] = c;
    else 
      // Nonstandard nuleotide (masked, etc). Leave as is.
      rev[s] = seq[len-s-1];
  }
 }

bool bfQuery(unsigned int hash_vector[NUM_HASH],unsigned char *bloom1,int nBloom, int minMult)
{		
	unsigned int bit,index,position;		

	for(int j = 0; j < NUM_HASH; j++) //15 hash functions
	{
		bit  = hash_vector[j] % char_size; //0 - 7
		index = hash_vector[j] / char_size;
		
		position = (index * char_size + bit) * minMult + nBloom; 
		
		//re-caculate new index and bit based on position
		bit = position % char_size;
		index = position / char_size;

		if (( bloom1[index] & bit_mask[bit]) != bit_mask[bit])
		{
			return false;
		}
	}
    
    return true; 
}

void bfInsert(unsigned int hash_vector[NUM_HASH],unsigned char *bloom1,int nBloom, int minMult)
{	
	unsigned int bit,index,position;

	for(int i= 0; i < NUM_HASH; i++) 
	{	
		bit  = hash_vector[i] % char_size;
		index = hash_vector[i] / char_size;
		
		position = (index * char_size + bit) * minMult + nBloom; 
		
		//re-caculate new index and bit based on position
		bit = position % char_size;
		index = position / char_size;

		bloom1[index] |= bit_mask[bit];
	}
}


void bf_insert(char *tuple,int table_size, unsigned char *bloom1, hash_function hash_function_list[],int tuple_len)
{	
	unsigned int hash;
	for(int i= 0; i < NUM_HASH; i++) 
	{
		 hash = hash_function_list[i](tuple,tuple_len) % (table_size * char_size);
		bloom1[hash / char_size] |= bit_mask[hash % char_size];
	}
}

//Check whether bloom filter contains "string key"
 bool bf_query(char *key, unsigned int table_size,unsigned char *bloom1,hash_function hash_function_list[],int tuple_len)
{	
	unsigned int hash, bit, index,len;
	unsigned char bloom;

	char str[TUPLE_SIZE+2];
	
	strncpy(str, key,TUPLE_SIZE+1);
	str[TUPLE_SIZE+1]=0;

	for(int j = 0; j < NUM_HASH; j++) //15 hash functions
	{
			 
		hash = hash_function_list[j](str,tuple_len) % (table_size * char_size);

		bit  = hash % char_size;
		index = hash / char_size ;    
		bloom = bloom1[index];

		if ((bloom & bit_mask[bit]) != bit_mask[bit])
		{
			return false;
		}
	}
    
    return true; 
}


  void dnas_RemoveRepeats(DNASequence *dnas) {
	int i;
	for (i = 0; i < dnas->length; i++) {
		dnas->seq[i] = unmasked_nuc[dnas->seq[i]];
	}
}
 
 char UnmaskedValue(char numericNuc) { 
    if (numericNuc < 0) 
      return (-numericNuc -1) % 4;
    else
      return numericNuc;
 }


int GetSeq(FILE *in, DNASequence *sequence)
{
	int MAX_REC_LEN=500;
	int newbuflen;
	char p;


	if (feof(in))
		return 0;
	
	p = fgetc(in);	
	if(p == '\0' || p== '\n'|| p == ' ')
	{
		return 0;
	}

	if( p == '>') 
	{			
		fgets(sequence->namestr, MAX_REC_LEN, in); 

		//remove newline feed
		newbuflen = strlen(sequence->namestr);
		if (sequence->namestr[newbuflen - 1] == '\n') 
			sequence->namestr[newbuflen - 1] = '\0';

		sequence->namestr[NAMESTR_LEN-1]='\0';
	 }
		
	fgets(sequence->seq, MAX_REC_LEN, in);

	//remove newline feed	
	newbuflen = strlen(sequence->seq);
		
	if ((sequence->seq[newbuflen - 1] == '\n') || (sequence->seq[newbuflen - 1] == '#') )
		sequence->seq[newbuflen - 1] = '\0';	

	sequence->length = strlen(sequence->seq);
	dnas_RemoveRepeats(sequence);
	
	return 1; 
}

	

	

void PrintUsage() {
	printf("fixErrorsVoting   Fix errors in reads using spectral alignment with \n");
	printf("a voting approach instead of a dynamic programming. \n");
	printf("This has the benefit of being able to fix point \n");
	printf(" mutations or single indels in very small reads (<25 nt) \n");
	printf("assuming there is no more than one or two errors per read \n");
	printf("Usage: fixErrorsVoting seqFile spectrumFile tupleSize outputFile [options] \n");
	printf("-minMult  m  Only consider tuples with multiplicity above m \n");
	printf("to be solid.\n");
	printf("-minVotes v  Require at least 'v' separate votes to fix any position.\n");
	printf("A vote is cast for a position p, nucleotide n, if a \n");
	printf("change at (p,n) makes a tuple t change from below m \n");
	printf("to above.\n");
	printf(" -maxTrim  x  Trim at most x nucleotides off the ends of a read. If \n");
	printf("more than x nucleotides need to be trimmed, the read is unfixable.\n");
	printf("  -deletions   Search for single deletions. \n");
	printf("   -insertions  Search for single insertions.\n");
	printf("    -search s    Try up to 's' changes to make a tuple solid. \n");
	printf("    -compare file For benchmarking purposes, the correct reads are given \n");
	printf("in file 'file'.  Read those and copare the results.\n");
	printf("     -discardFile file Print all reads that do not pass the threshold to 'file'\n");

	printf("    -map mapFile Print portions of retained reads to 'mapFile'.\n");
	printf("     -spectrum [concise|full].  Use either a concise or full spectrum. \n");
	printf("The concise spectrum must be on words of size less than 16 \n");
	printf("and resets all multiplicities greater than 3 to 3. \n");
	printf("The full spectrum may be of any length, and stores \n");
	printf("the exact multiplicity of all k-mers.\n");
}
                                                                                                               utils.h                                                                                             0000664 0000765 0000765 00000001455 11167253302 012552  0                                                                                                    ustar   haixiang                        haixiang                                                                                                                                                                                                               #ifndef UTILS_H_
#define UTILS_H_

#include "FixErrorsVoting.h"

extern "C" void MakeRC(char* seq, int len, char* &rev);
extern "C" bool bfQuery(unsigned int hash_vector[NUM_HASH],unsigned char *bloom1,int nBloom, int minMult);
extern "C" void bfInsert(unsigned int hash_vector[NUM_HASH],unsigned char *bloom1,int nBloom, int minMul);
extern "C" void bf_insert(char *tuple,int table_size, unsigned char *bloom1, hash_function hash_function_list[],int tuple_len);
extern "C" bool bf_query(char *key, unsigned int table_size,unsigned char *bloom1,hash_function hash_function_list[],int tuple_len);

extern "C" void PrintUsage();
extern "C" void dnas_RemoveRepeats(DNASequence *dnas);
extern "C" char UnmaskedValue(char numericNuc);
extern "C" int GetSeq(FILE *in, DNASequence *sequence);


#endif
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   