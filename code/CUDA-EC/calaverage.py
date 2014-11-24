#! /usr/bin/python

import re
import commands

(a, filename) = commands.getstatusoutput("ls CUDA-EC.o*")
f = open(filename, 'r')
text = f.read()

pattern = "The run time for fixing error in GPU is: [\d\.]* secs\."
pattern_small = "[\d\.]+"
sum = 0.0
for str in re.findall(pattern, text):
	sum += (float) (re.findall(pattern_small, str)[0])

print "average run time of 5 runs for fixing error in GPU is: "+repr(sum/5)+"secs"

