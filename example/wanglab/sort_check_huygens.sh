#!/bin/bash

basepath="/stanley/WangLab/Data/Processed"
ls $basepath

# Ask user what the sample name is:
read -rp "Please enter name of sample directory: " sample
if [ ! -d "$basepath/$sample" ]; then
  echo "$basepath/$sample does not exist. Exiting program..."
  exit 0
fi
printf 'Navigating to %s/%s\n' "$basepath" "$sample"
cd $basepath/$sample
if compgen -G "task*" > /dev/null; then
  rm *.log
fi

# Ask user how many positions are expected in each round:
read -rp "Please enter number of positions or tiles in each round: " npos

# Ask user for input on which round they want to check:
#echo "Enter round number to sort and check (or press Ctrl-C to exit):"
while IFS= read -rp "Enter round number to sort and check (or press Ctrl-C to exit): " r_num; do
  # First restructure existing files into individual position directories
  printf 'Restructuring files in round %s...\n' "$r_num"
  round_dir="round$r_num"
#  rm $round_dir/task*
#  rm $round_dir/sched*
  find $round_dir -mindepth 1 -maxdepth 1 -type f -name '*Position*' -print0 | 
  grep -a -o 'Position[0-9]\+' | 
  sort | uniq | while read pos; do # match position names
    mkdir -p $round_dir/$pos # create directory for each position 
    find $round_dir -mindepth 1 -maxdepth 1 -type f | # get names of unsorted files
    sed "s/$round_dir\///g" | # remove round from file name
    grep $pos | while read file; do # match position 
      mv "$round_dir/$file" "$round_dir/$pos/$file"
    done
  done 
  printf 'Finished restructuring round %s\n' "$r_num"

  # Check for missing positions
  ndirs=`find $round_dir -mindepth 1 -maxdepth 1 -type d | wc -l | sed 's/ //g'`
  if [ "$ndirs" -eq $npos ]; then
    echo "All $npos positions for round $r_num found"
  elif [ "$ndirs" -gt $npos ]; then
    echo "WARNING: $ndirs positions found when $npos positions expected"
  else
    # Find missing dirs
    echo "MISSING POSITIONS: $ndirs positions found when $npos positions expected. The following positions are missing (assuming first position is 1):" 
    expdirs=($(seq 1 $npos))
    obsdirs=(`find $round_dir -mindepth 1 -maxdepth 1 -type d -name 'Position*' -print0 | grep -a -o 'Position[0-9]\+' | sort | sed 's/Position//g' | sed 's/^0*//g'`) 
    echo ${expdirs[@]} ${obsdirs[@]} | tr ' ' '\n' | sort --version-sort | uniq -u
  fi

  # Check for missing files in each position
  read -rp "Enter number of channels in this round to check if all files present (enter 0 to skip): " n_chs
  if [ $n_chs -ne 0 ]; then
    for pos in `find $round_dir -mindepth 1 -maxdepth 1 -type d | sort`; do
      for i in $(eval echo {0..$((n_chs-1))}); do
        if ls $pos | grep -q "ch0$i"; then 
          continue
        else
          echo "Missing: $pos: ch0$i.tif"
        fi 
      done
    done 
   
    echo Finished checking all positions
  fi

done


