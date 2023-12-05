#!/bin/bash
ppath="/stanley/WangLab/jiahao/Github/starfinder/test/TEMPOmap_Hela"
outpath="$ppath/mscript"
main_mscript="$ppath/TEMPOmap_Hela_run.m"

j=1
for i in $(seq -f "%03g" 351 420)
# cat $listpath|while read file
do
	echo "current_fov='Position$i'" > $outpath/task_$j".m"
	cat $main_mscript >> $outpath/task_$j".m"
	j=$((j+1))
done
