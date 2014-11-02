CUDA-EC: A Parallel Fast error correction tool for high-throughput short-reads DNA Sequence.

-------------
CUDA-EC v.1.0
-------------
Code base downlaoded from this paper
Haixiang Shi, Bertil Schmidt, Weiguo Liu, and Wolfgang M¨¹ller-Wittig: "A Parallel Algorithm for Error Correction in High-Throughput Short-Read Data on CUDA-enabled Graphics Hardware", Manuscript submitted 

-------------
Installation: 
-------------
There is a Makefile in this folder which should compile the code.
This code was initially written with CUDA 1.1, which is very old and depended on cutil.h (deprecated now). This library has been manually placed in the "common" directory now and linked with the code.

------
Usage: 
------
./CUDA-EC -f {inputfilename} -t {kmer size} -o {fixed-filename} -d {discarded-filename} -r {read_length}[-maxTrim {maximum_trim}] [-minVotes {minimum votes}] [-minMult {multiplicity}] [-search {num_error_to_fix]]

--------
Example:
--------
A sample fasta file SRR1552370.fasta is downloaded from NCBI in the input folder. In this sample, each read is of length 36. Assuming an error rate of 1% and our kmer size preference to be 20, run it using the following command:

./CUDA-EC  -f input/SRR1552370.fasta -t 20 -o output/SRR1552370.fasta.fixed -d output/SRR1552370.fasta.discards -r 36 -search 1

Run it using the "qsub ecJob.pbs" to avoid using the head node.

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

-t {kmer size}							--	length of kmer
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

CUDA-EC outputs two files in the output directry, one is the fixed file contains the fixed and trimmed reads, the another is the file contains the discarded reads. 


