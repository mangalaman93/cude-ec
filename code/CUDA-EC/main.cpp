/*
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
#include <string.h>

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

  int minMult=3;
  short doInsertion=0, doDeletion=0, doTrim=0, maxTrim=20;
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

  printf("Malloc reads array memory finished (%d MB) \n", ((sizeof(char)*arr_size)/(1024*1024)));

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

