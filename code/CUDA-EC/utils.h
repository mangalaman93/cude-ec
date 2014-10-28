#ifndef UTILS_H_
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
