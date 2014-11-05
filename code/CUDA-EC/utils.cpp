
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
