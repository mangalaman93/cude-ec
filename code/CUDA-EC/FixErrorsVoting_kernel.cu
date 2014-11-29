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
  //char str[TUPLE_SIZE+1];
  char *str = key;

  //_strncpy_(str, key,TUPLE_SIZE);
  //str[TUPLE_SIZE]=0;


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
  //char str[TUPLE_SIZE+1];
  char *str= key;

  //_strncpy_(str, key,TUPLE_SIZE);
  //str[TUPLE_SIZE]=0;

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

/*
Written by Aman Mangal, Chirag Jain on Nov 24, 2014
Algorithm: 1 warp -> 1 read
*/

////////////////////////////////////////////////////////////////////////////////
//! Fix two errors, step 1 kernel function
//! @param d_reads_arr  input data in global memory
//! @param d_param  input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void fix_errors1_warp_copy(char *d_reads_arr,Param *d_param)
{
  short numSearch=1;
  nextNuc['G'] = 'A'; nextNuc['A'] = 'C'; nextNuc['C'] = 'T'; nextNuc['T'] = 'G';

  int c_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int w_tid = threadIdx.x & (WARPSIZE - 1);
  int round = 0;

  //# changing total number of threads
  int total_thread = BLOCK * THREAD / WARPSIZE;
  int discardSeq=0;
  int trimStart=0, trimEnd=0;

  int chunk_bound = (total_thread < MAX_READS_BOUND ? total_thread:MAX_READS_BOUND);
  round = d_param->NUM_OF_READS/chunk_bound + (d_param->NUM_OF_READS%chunk_bound == 0 ? 0:1);

  int maxPos[READ_LENGTH * 4],maxMod[READ_LENGTH * 4];
  
  // unsigned char votes[READ_LENGTH][4];
  unsigned char mutNuc, mutNuc2, prev, cur;
  __shared__ unsigned char votes_shared[WARPS_BLOCK*READ_LENGTH*4];
  unsigned char* votes = &votes_shared[READ_LENGTH*4*(threadIdx.x/WARPSIZE)];
  
  // int solid[READ_LENGTH];
  __shared__ bool solid_shared[READ_LENGTH*WARPS_BLOCK];
  bool *solid = &solid_shared[READ_LENGTH * (threadIdx.x/WARPSIZE)];
  
  int s,i,j,m,n,startPos, fixPos=-1,numFixed = 0,numChanges=0;
  short return_value = 0,flag = 0,flag1=1;

  // Cast votes for mutations
  int p,vp,mut;
  short numAboveThreshold = 0,newLength,len;
  short maxVotes = 0,allGood  = 1;
  int numTies = -1,pindex = 0, mod, pos,current_read_idx;

  //Access to shared memory
  extern __shared__ char buffer[];

  char *tempTuple, *read, *readsInOneRound_Warp = &buffer[(threadIdx.x/WARPSIZE)*WARPSIZE*(d_param->readLen + 2)];

  for(i=0;i<round;i++)
  {
    flag = 0;flag1=1;numFixed = 0;  numChanges=0; return_value = 0;discardSeq = 0;

    current_read_idx = c_tid/WARPSIZE + chunk_bound * i;

    //check if run out of reads
    current_read_idx = (current_read_idx > d_param->NUM_OF_READS ? 0:current_read_idx);

    //Place reads in the shared memory after every round
    //Computing start offset for this warp
    //Go till end offset for this warp
    //Fill the shared buffer while coalescing global memory accesses
    //Doing it at a warp level will remove the requirement of syncing threads
    //int startOffsetForThisWarp = ((threadIdx.x/WARPSIZE)*WARPSIZE + blockIdx.x*blockDim.x + chunk_bound * i)* (d_param->readLen + 2);
    //int endOffsetForThisWarp = min(\
    //    ((threadIdx.x/WARPSIZE + 1)*WARPSIZE + blockIdx.x*blockDim.x + chunk_bound * i)* (d_param->readLen + 2),\
    //    d_param->NUM_OF_READS * (d_param->readLen + 2));

    //for (int j= startOffsetForThisWarp; j< endOffsetForThisWarp; j += WARPSIZE)
    //{
    //  if(j+threadIdx.x%WARPSIZE < endOffsetForThisWarp)
    //    readsInOneRound_Warp[j-startOffsetForThisWarp + threadIdx.x%WARPSIZE] = d_reads_arr[j+threadIdx.x%WARPSIZE];
    //}

    //read = &readsInOneRound_Warp[(threadIdx.x % WARPSIZE) * (d_param->readLen + 2)];
    read = &d_reads_arr[current_read_idx*(READ_LENGTH + 2)];

    //take 1 read per thread
    //read = &d_reads_arr[current_read_idx*(READ_LENGTH + 2)];

    // get length of this read
    len = read[READ_LENGTH + 1];

    if (!PrepareSequence(read)) {
if (w_tid == 0)
      discardSeq = 1;
    }
    else
    {
if (w_tid == 0)
{
      numFixed = 0; fixPos = -1;
}
      do{
        if(__any(flag))
          break;
        else{
if (w_tid == 0)
{
          if (fixPos > 0)
            startPos = fixPos;
          else
            startPos = 0;
}

          //# parallelizing this loop
          for(m=w_tid; m<READ_LENGTH; m+=WARPSIZE)
          {
                votes[m*4+0] = 0;
                votes[m*4+1] = 0;
                votes[m*4+2] = 0;
                votes[m*4+3] = 0;
          }
          for(m=w_tid; m<READ_LENGTH; m+=WARPSIZE)
          {
              solid[m] = 0;
          }

          // for (m = 0; m < READ_LENGTH; m++) {
          //   for (int n = 0; n < 4; n++)
          //     //votes[threadIdx.x][m][n] = 0;
          //     votes_2d(m,n) = 0;
          // }

          // for(m=0;m<READ_LENGTH;m++)
          //   solid[m] = 0;

if (w_tid == 0)
{
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
                      votes_2d(vp + p,unmasked_nuc_index[mutNuc])++;

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

          for (p = 0; p < len - d_param->tupleSize + 1; p++ ) {
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
                if (votes_2d(p,m) > d_param->minVotes)
                  numAboveThreshold++;

                if (votes_2d(p,m) >= maxVotes)
                  maxVotes = votes_2d(p,m);
              }
            }

            pindex = 0;numTies = -1;

            // Make sure there aren't multiple possible fixes
            for (p = 0; p < len; p++){
              for (m = 0; m < 4; m++){
                if (votes_2d(p,m) == maxVotes){
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
    }

if (w_tid == 0)
{
    if (discardSeq) {
      read[READ_LENGTH] = 'D'; //F fixed, D: not fixed, discard
    }
    else {
      read[READ_LENGTH] = 'F'; //F fixed, D: not fixed, discard
    }
}
    ////Save back results to global memory
    //for (int j= startOffsetForThisWarp; j< endOffsetForThisWarp; j += WARPSIZE)
    //{
    //  if(j+threadIdx.x%WARPSIZE < endOffsetForThisWarp)
    //    d_reads_arr[j+threadIdx.x%WARPSIZE] = readsInOneRound_Warp[j-startOffsetForThisWarp + threadIdx.x%WARPSIZE];
    //}

    __syncthreads();
  }
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

  //Access to shared memory
  extern __shared__ char buffer[];

  char *tempTuple, *read, *readsInOneRound_Warp = &buffer[(threadIdx.x/WARPSIZE)*WARPSIZE*(d_param->readLen + 2)];

  for(i=0;i<round;i++)
  {
    flag = 0;flag1=1;numFixed = 0;	numChanges=0;	return_value = 0;discardSeq = 0;

    current_read_idx = c_tid + chunk_bound * i;

    //check if run out of reads
    current_read_idx = (current_read_idx > d_param->NUM_OF_READS ? 0:current_read_idx);

    //Place reads in the shared memory after every round
    //Computing start offset for this warp
    //Go till end offset for this warp
    //Fill the shared buffer while coalescing global memory accesses
    //Doing it at a warp level will remove the requirement of syncing threads
    int startOffsetForThisWarp = ((threadIdx.x/WARPSIZE)*WARPSIZE + blockIdx.x*blockDim.x + chunk_bound * i)* (d_param->readLen + 2);
    int endOffsetForThisWarp = min(\
        ((threadIdx.x/WARPSIZE + 1)*WARPSIZE + blockIdx.x*blockDim.x + chunk_bound * i)* (d_param->readLen + 2),\
        d_param->NUM_OF_READS * (d_param->readLen + 2));

    for (int j= startOffsetForThisWarp; j< endOffsetForThisWarp; j += WARPSIZE)
    {
      if(j+threadIdx.x%WARPSIZE < endOffsetForThisWarp)
        readsInOneRound_Warp[j-startOffsetForThisWarp + threadIdx.x%WARPSIZE] = d_reads_arr[j+threadIdx.x%WARPSIZE];
    }

    read = &readsInOneRound_Warp[(threadIdx.x % WARPSIZE) * (d_param->readLen + 2)];

    //take 1 read per thread
    //read = &d_reads_arr[current_read_idx*(READ_LENGTH + 2)];

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

    //Save back results to global memory
    for (int j= startOffsetForThisWarp; j< endOffsetForThisWarp; j += WARPSIZE)
    {
      if(j+threadIdx.x%WARPSIZE < endOffsetForThisWarp)
        d_reads_arr[j+threadIdx.x%WARPSIZE] = readsInOneRound_Warp[j-startOffsetForThisWarp + threadIdx.x%WARPSIZE];
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
