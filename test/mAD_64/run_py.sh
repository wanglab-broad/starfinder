#!/bin/bash -l
ppath="/stanley/WangLab/jiahao/Github/RIBOmap/cluster-clustermap-3"
#$ -o /stanley/WangLab/jiahao/Github/RIBOmap/cluster-clustermap-3/log/qsub_log_o.$JOB_ID.$TASK_ID
#$ -e /stanley/WangLab/jiahao/Github/RIBOmap/cluster-clustermap-3/log/qsub_log_e.$JOB_ID.$TASK_ID

source "/broad/software/scripts/useuse"
reuse Anaconda3
source activate /stanley/WangLab/envs/clustermap
now=$(date +"%T")
echo "Current time : $now"
position=$(sed "${SGE_TASK_ID}q;d" /stanley/WangLab/jiahao/Github/RIBOmap/cluster-clustermap-3/code/position.txt)
position=$(echo $position | tr -d '\r')
python $ppath/code/starmap_clustermap_tile.py $position

echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
