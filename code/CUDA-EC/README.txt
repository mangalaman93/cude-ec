CUDA-EC: A Parallel Fast error correction tool for high-throughput short-reads DNA Sequence.

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
Haixiang Shi, Bertil Schmidt, Weiguo Liu, and Wolfgang M¨¹ller-Wittig: "A Parallel Algorithm for Error Correction in High-Throughput Short-Read Data on CUDA-enabled Graphics Hardware", Manuscript submitted 

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
 




