#!/bin/bash -l
# Generate stitchlinks for DAPI stitching 

sample=$1
#tissue=`echo $sample | cut -f1 -d '_'`
orderlist=$2
segmethod=$3
outpath=$4
segoutpath=$5

mkdir -p $outpath 

i=1
for j in `cat $orderlist`
do 
  j=`printf "%03d" $j`
  if [ "$j" = "0" ] || [ ! -s $segoutpath/Position$j/max_rotated_dapi.tif ] 
  then
    ln -sf ../blank.tif $outpath/tile_$i.tif
  else
    ln -sf ../../03_segmentation/$segmethod/Position$j/max_rotated_dapi.tif $outpath/tile_$i.tif 
  fi
  i=$((i+1))
done
