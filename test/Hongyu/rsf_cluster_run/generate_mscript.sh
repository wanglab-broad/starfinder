#!/bin/bash
ppath="/stanley/WangLab/jiahao/Github/starfinder/test/Hongyu/rsf_cluster_run"
outpath="$ppath/mscript"
main_mscript="$ppath/Hongyu_spleen_run.m"

j=1
for i in $(seq -f "%03g" 1 1078)
# cat $listpath|while read file
do
	echo "current_fov='Position$i'" > $outpath/task_$j".m"
	cat $main_mscript >> $outpath/task_$j".m"
	j=$((j+1))
done