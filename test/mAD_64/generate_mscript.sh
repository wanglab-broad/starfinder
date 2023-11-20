#!/bin/bash
ppath="/stanley/WangLab/jiahao/Github/starfinder/test/mAD_64"
outpath="$ppath/mscript"
main_mscript="$ppath/mAD_64_run.m"

j=1
for i in $(seq -f "%03g" 1 56)
# cat $listpath|while read file
do
	echo "current_fov='tile_$i'" > $outpath/task_$j".m"
	cat $main_mscript >> $outpath/task_$j".m"
	j=$((j+1))
done
