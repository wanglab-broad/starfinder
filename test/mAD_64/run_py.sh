#!/bin/bash -l
ppath="/stanley/WangLab/jiahao/Github/starfinder/test/mAD_64"
#$ -o qsub_log/output/o.$JOB_ID.$TASK_ID
#$ -e qsub_log/error/e.$JOB_ID.$TASK_ID

source "/broad/software/scripts/useuse"
reuse Anaconda3
source activate /stanley/WangLab/envs/clustermap
now=$(date +"%T")
echo "Current time : $now"
fov=$(sed "${SGE_TASK_ID}q;d" /stanley/WangLab/jiahao/Github/starfinder/test/mAD_64/fov.txt)
fov=$(echo $fov | tr -d '\r')
python $ppath/run_clustermap.py $fov

echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
