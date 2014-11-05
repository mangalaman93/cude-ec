METRICS="tex_cache_hit_rate,\
tex_cache_throughput,\
l2_texture_read_hit_rate,\
l2_texture_read_throughput,\
tex_cache_transactions,\
tex_utilization"

if [[ -z "$1" ]]
then
	FILENAME=SRR006331
else
	FILENAME=$1
fi

COMMAND="nvprof --print-gpu-trace --metrics ${METRICS} ./CUDA-EC -f /work/alurugroup/chirag/CUDA-EC/input/${FILENAME}.fasta -t 21 -o /work/alurugroup/chirag/CUDA-EC/output/${FILENAME}.fasta.fixed -d /work/alurugroup/chirag/CUDA-EC/output/${FILENAME}.fasta.discards -r 36 -search 1 -minMult 5"

echo ${COMMAND}

${COMMAND}
