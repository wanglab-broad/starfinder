#!/bin/bash -l
ppath="/stanley/WangLab/jiahao/Github/starfinder/test/Hongyu/rsf_cluster_run"
#$ -o qsub_log/output/o.$JOB_ID.$TASK_ID
#$ -e qsub_log/error/e.$JOB_ID.$TASK_ID

source "/broad/software/scripts/useuse"
reuse Matlab
now=$(date +"%T")
echo "Current time : $now"
command="run('$ppath/mscript/task_"$SGE_TASK_ID"');exit;"
matlab -nodisplay -nosplash -nodesktop -r $command

echo "Finished"
now=$(date +"%T")
echo "Current time : $now"
