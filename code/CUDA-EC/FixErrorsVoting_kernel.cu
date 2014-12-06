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
#define votes_2d(m,n) votes[(m)*4+(n)]

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

__constant__ char nuc_char[4] = {'G', 'A', 'C', 'T'};

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
  4,4,4,4,4,4,4,1,4,2,   // 90 99
  4,4,4,0,4,4,4,4,4,4,   // 100 109
  4,4,4,4,4,4,3,4,4,4,   // 110 119
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

__device__ int _toupper_(int ch)
{
  if ((unsigned int)(ch - 'a') < 26u )
    ch += 'A' - 'a';
  return ch;
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

__device__ int lstspct_FindTuple_With_Copy(char* tuple, int numTuples)
{
  unsigned int hash, bit, index;
  unsigned char bloom;

  unsigned int b = 378551;
  unsigned int a = 63689;
  numTuples *= BLOOM_SIZE;

  //_RSHash_
  hash = 0;
  for(unsigned i=0; i<TUPLE_SIZE; i++)
  {
    hash = hash * a + (tuple[i]);
    a    = a * b;
  }

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);
  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }


  //_JSHash_
  hash = 1315423911;
  for(unsigned i = 0; i < TUPLE_SIZE; i++)
  {
    hash ^= ((hash << 5) + (tuple[i]) + (hash >> 2));
  }

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);
  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }


  //_PJWHash_
  unsigned int ThreeQuarters = (unsigned int)(((unsigned int)(sizeof(unsigned int) * 8)  * 3) / 4);
  unsigned int HighBits = (unsigned int)(0xFFFFFFFF) << (sizeof(unsigned int) * 7);
  hash = 0;
  a = 0;

  for(unsigned i = 0; i < TUPLE_SIZE; i++)
  {
    hash = (hash << sizeof(unsigned int)) + (tuple[i]);

    if((a = hash & HighBits)  != 0)
      hash = (( hash ^ (a >> ThreeQuarters)) & (~HighBits));
  }

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);
  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }

  //_ELFHash_
  hash = 0;
  a = 0;
  for(unsigned i = 0; i < TUPLE_SIZE; i++)
  {
    hash = (hash << 4) + (tuple[i]);
    if((a = hash & 0xF0000000L) != 0)
      hash ^= (a >> 24);
    hash &= ~a;
  }

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);  ;

  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }

  //_BKDRHash_
  hash=0;a=131;
  for(unsigned i = 0; i < TUPLE_SIZE; i++)
    hash = (hash * a) + (tuple[i]);

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);
  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }

  //_SDBMHash_
  hash=0;

  for(unsigned i = 0; i < TUPLE_SIZE; i++)
    hash = (tuple[i]) + (hash << 6) + (hash << 16) - hash;

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);

  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }


  //_DJBHash_
  hash = 5381;
  for(unsigned i = 0; i < TUPLE_SIZE; i++)
    hash = ((hash << 5) + hash) + (tuple[i]);

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);

  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }

  //_DEKHash_
  hash = TUPLE_SIZE;
  for(unsigned i = 0; i < TUPLE_SIZE; i++)
    hash = ((hash << 5) ^ (hash >> 27)) ^ (tuple[i]);

  hash = hash % (numTuples * _char_size_);
  bit  = hash & (_char_size_-1);
  index = hash >> 3; // _char_size_ ;
  bloom = tex1Dfetch( tex, index);

  if ((bloom & _bit_mask_[bit]) != _bit_mask_[bit])
  {
    return -1;
  }

  return 1;
}

//Check each char inside this read, only "A/C/T/G" allowed in the fasta file
__device__ int PrepareSequence(char *read)
{
  for (unsigned p = 0; p < READ_LENGTH; p++ )
  {
    read[p] = _toupper_(read[p]);
    if (!(read[p] == 'A' ||
          read[p] == 'C' ||
          read[p] == 'T' ||
          read[p] == 'G'))
    {
      return 0;
    }
  }
  return 1;
}

__device__ int d_strTpl_Valid(char *st)
{
  if (st == NULL)
    return 0;
  else
  {
    for (unsigned i = 0; i < TUPLE_SIZE; i++) {
      if (numeric_nuc_index[st[i]] >= 4)
      {
        return 0;
      }
    }
  }
  return 1;
}

//check whether the read is solid or not, by examine each tuple in this read, whether can be found or not in
//all the string tuple list
__device__ int CheckSolid(char *seq, int tupleSize, int numTuples){
  for (unsigned p = 0; p < READ_LENGTH - tupleSize +1; p++ ){
    if (lstspct_FindTuple_With_Copy(seq+p, numTuples)==-1) {
      return 0;
    }
  }

  return 1;
}

__device__ int SolidSubsequence(char *seq, int tupleSize, int &seqStart, int &seqEnd, int numTuples) {
  for (unsigned i = seqStart; i < seqEnd - tupleSize + 1; i++)
  {
    if(lstspct_FindTuple_With_Copy(&seq[i], numTuples) == -1) {
      return 0;
    }
  }

  return 1;
}

__device__ int TrimSequence(char *seq, int tupleSize, int numTuples,int maxTrim)
{
  int seqStart, seqEnd;
  int flag = 1;

  //get length of this read
  int len = seq[READ_LENGTH + 1];

  for(seqStart = 0; seqStart < len - tupleSize + 1; seqStart++ )
  {
    if (lstspct_FindTuple_With_Copy(&seq[seqStart],numTuples) != -1) {
      break;
    }
  }

  unsigned i = seqStart + 1;
  for (; i < len - tupleSize + 1; i++ )
  {
    if(lstspct_FindTuple_With_Copy(&seq[i], numTuples) == -1) {
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

  if (seqStart > maxTrim) {
    flag = 0;
  } else if (len - seqEnd > maxTrim) {
    flag = 0;
  } else if(SolidSubsequence(seq, tupleSize, seqStart, seqEnd,numTuples) == 0) {
    flag = 0;
  } else {
    int newLength = seqEnd - seqStart + 1;

    for (int s = 0; s < newLength; s++ ) {
      seq[s] = seq[s + seqStart];
    }

    len = newLength -1;
  }

  seq[READ_LENGTH + 1] = len;
  return flag;
}

////////////////////////////////////////////////////////////////////////////////
//! Fix two errors, step 1 kernel function
//! @param d_reads_arr  input data in global memory
//! @param d_param  input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void fix_errors1_warp_copy(char *d_reads_arr,Param *d_param)
{
  nextNuc['G'] = 'A'; nextNuc['A'] = 'C'; nextNuc['C'] = 'T'; nextNuc['T'] = 'G';

  int w_tid = threadIdx.x & (WARPSIZE - 1);
  int w_id = threadIdx.x >> 5;

  //# changing total number of threads
  int chunk_bound = BLOCK * THREAD / WARPSIZE;
  int discardSeq=0;
  int round = d_param->NUM_OF_READS/chunk_bound + (d_param->NUM_OF_READS%chunk_bound == 0 ? 0:1);
  int maxPos[2],maxMod[2];
  
  unsigned char mutNuc;
  __shared__ unsigned char votes_shared[WARPS_BLOCK*READ_LENGTH*4];
  unsigned char* votes = &votes_shared[READ_LENGTH*4*(threadIdx.x/WARPSIZE)];
  __shared__ int startPos[WARPS_BLOCK];
  
  int fixPos=-1,numFixed = 0,numChanges=0;
  short return_value = 0,flag = 0;

  // Cast votes for mutations
  short numAboveThreshold = 0,len;
  short maxVotes = 0,allGood  = 0;
  int pindex = 0;

  //Access to shared memory
  extern __shared__ char buffer[];

  char *read, *readsInOneRound_Warp = &buffer[w_id * (d_param->readLen + 2)];

  for(unsigned i=0;i<round;i++)
  {
    flag = 0;numFixed = 0;  numChanges=0; return_value = 0;discardSeq = 0;

    //Place reads in the shared memory after every round
    //Computing start offset for this warp
    //Go till end offset for this warp
    //Fill the shared buffer while coalescing global memory accesses
    //Doing it at a warp level will remove the requirement of syncing threads
    int startOffsetForThisWarp = ((w_id) + blockIdx.x*WARPS_BLOCK + chunk_bound * i)* (d_param->readLen + 2);
    int endOffsetForThisWarp = min(\
          ((w_id + 1) + blockIdx.x*WARPS_BLOCK + chunk_bound * i)* (d_param->readLen + 2),\
          d_param->NUM_OF_READS * (d_param->readLen + 2));

    for (int j= startOffsetForThisWarp + w_tid; j< endOffsetForThisWarp; j += WARPSIZE)
    {
      if(j  < endOffsetForThisWarp)
        readsInOneRound_Warp[j-startOffsetForThisWarp] = d_reads_arr[j];
    }

    read = readsInOneRound_Warp;

    // get length of this read
    len = read[READ_LENGTH + 1];

    // flag variable, PrepareSequence function
    pindex = 1;
    for(unsigned p = w_tid; p < READ_LENGTH; p+=WARPSIZE )
    {
      read[p] = _toupper_(read[p]);
      if (!(read[p] == 'A' ||
            read[p] == 'C' ||
            read[p] == 'T' ||
            read[p] == 'G'))
      {
        pindex = 0;
      }
    }

    if(__any(pindex == 0))
    {
      if(w_tid == 0 )
      {
        discardSeq = 1;
      }
    } else
    {
      if (w_tid == 0)
      {
        numFixed = 0;
        fixPos = -1;
      }

      do
      {
        if(__any(flag))
        {
          break;
        } else
        {
          if (w_tid == 0)
          {
            if (fixPos > 0)
            {
              startPos[w_id] = fixPos;
            } else
            {
              startPos[w_id] = 0;
            }
          }

          //# parallelizing this loop
          for(unsigned m=w_tid; m<READ_LENGTH; m+=WARPSIZE)
          {
                votes[m*4+0] = 0;
                votes[m*4+1] = 0;
                votes[m*4+2] = 0;
                votes[m*4+3] = 0;
          }
          
          allGood = 0;
          char str[READ_LENGTH];
          _strncpy_(str, read, READ_LENGTH);
          for(unsigned p = startPos[w_id]; p < len - d_param->tupleSize + 1; p++ )
          {
            char* tempTuple = &str[p];
            if (d_strTpl_Valid(tempTuple))
            {
              if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
              {
                allGood++; //solid[p] = 1;
              } else
              {
                for (unsigned vp = w_tid; vp < d_param->tupleSize; vp+=WARPSIZE)
                {
                  mutNuc = nextNuc[tempTuple[vp]];
                  tempTuple[vp] = mutNuc;

                  if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
                    votes_2d(vp + p,unmasked_nuc_index[mutNuc])++;

                  mutNuc = nextNuc[mutNuc];
                  tempTuple[vp] = mutNuc;

                  if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
                    votes_2d(vp + p,unmasked_nuc_index[mutNuc])++;

                  mutNuc = nextNuc[mutNuc];
                  tempTuple[vp] = mutNuc;

                  if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
                    votes_2d(vp + p,unmasked_nuc_index[mutNuc])++;

                  mutNuc = nextNuc[mutNuc];
                  tempTuple[vp] = mutNuc;
                }
              }
            }
          }
          
          if (w_tid==0)
          {
            ++numFixed;

            //////////////////////fix sequence based on voting in previous step//////////////
            fixPos = 0;numAboveThreshold = 0;maxVotes = 0;

            if (allGood == len-d_param->tupleSize+1)
              // no need to fix this sequence
              return_value =  1;
            else
            {
              for (unsigned p = 0; p < len; p++){
                for (unsigned m = 0; m < 4; m++){
                  if (votes_2d(p,m) > d_param->minVotes)
                    numAboveThreshold++;

                  if (votes_2d(p,m) >= maxVotes)
                    maxVotes = votes_2d(p,m);
                }
              }

              pindex = 0;

              // Make sure there aren't multiple possible fixes
              for (unsigned p = 0; p < len; p++){
                for (unsigned m = 0; m < 4; m++){
                  if (votes_2d(p,m) == maxVotes){
                    maxPos[pindex] = p;
                    maxMod[pindex] = m;
                    pindex++;
              		  if (pindex > 1) {
              		    m = 4;
              		    p = len;
              		  }
                  }
                }
              }

              if (numAboveThreshold > 0 ){
                if (pindex < 2 || (pindex > 1 && maxPos[0] != maxPos[1])){
                  // Found at least one change to the sequence
                  if (pindex>0) {
                    fixPos = maxPos[0];

                    if (maxMod[0] < 4){
                      read[maxPos[0]] = nuc_char[maxMod[0]];
                    }
                  }
                  if( CheckSolid(read,d_param->tupleSize,d_param->numTuples))
                    return_value = 1;
                  else{
                    //reset
                    return_value = 0;
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
        }
      } while (__any(fixPos > 0));
      /////////////////////////end of solidify////////

      if (w_tid == 0)
      {
        if (numChanges != 0){
          if (numChanges > d_param->maxMods)
            discardSeq = 1;
          else
            discardSeq = 0;
        } else
        {
          if( d_param->numSearch == 2)
          {
            //removed trim in fix error1
            discardSeq = 1;
          } else
          {
            // Find the locations of the first solid positions.
            if (d_param->doTrim)
            {
              if(TrimSequence(read, d_param->tupleSize, d_param->numTuples,d_param->maxTrim)){
                // If there is space for one solid tuple (trimStart < trimEnd - ts+1)
                // and the subsequence between the trimmed ends is ok, print the
                // trimmed coordinates.
                discardSeq = 0;
              } else
                discardSeq = 1;
            } else 
            {
              discardSeq = 1;
            }
          }
        }
      }
    }

    if (w_tid == 0)
    {
      if (discardSeq)
      {
        read[READ_LENGTH] = 'D'; //F fixed, D: not fixed, discard
      } else
      {
        read[READ_LENGTH] = 'F'; //F fixed, D: not fixed, discard
      }
    }

    //Save back results to global memory
    for (int j= startOffsetForThisWarp + w_tid; j< endOffsetForThisWarp; j += WARPSIZE)
    {
      if(j < endOffsetForThisWarp)
        d_reads_arr[j] = readsInOneRound_Warp[j-startOffsetForThisWarp];
    }
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
              if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
                solid[p] = 1;
              else{
                for (vp = 0; vp < d_param->tupleSize-1; vp++)
                {

                  mutNuc = nextNuc[read[p + vp]];
                  read[p + vp] = mutNuc;

                  for (mut = 0; mut < 3; mut++ )
                  {

                    tempTuple = &read[p];

                    if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
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
                        if (lstspct_FindTuple_With_Copy(tempTuple, d_param->numTuples) != -1)
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
          if(TrimSequence(read, d_param->tupleSize, d_param->numTuples,d_param->maxTrim)){
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

