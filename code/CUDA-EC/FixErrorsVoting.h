#ifndef FIXERRORSVOTING_H_
#define FIXERRORSVOTING_H_

#include "hash_function.h"

//Must define read length and tuple size as it used in kernel, which cannot be dynamic allocate memory. 
//In char[READ_LENGTH+1]
#define READ_LENGTH 36
#define TUPLE_SIZE 20

//#define NUM_OF_READS 1153738 // NC000913: 9279350//mw2:3857879 //NC007146:3828980 //NC001137:1153738 //NC001139: 2181894
//helicobacter,11628131

#define NUM_HASH 8 //number of hash function
#define BLOOM_SIZE 8

//Block and Thread number used in Kernel
#define BLOCK 256
#define THREAD 256

#define WARPSIZE 32
#define WARPS_BLOCK (BLOCK/WARPSIZE)

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
