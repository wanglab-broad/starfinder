# Shell script for creating order list of positions

# Parse arguments
tissue=$1
maf=$tissue".maf"
sample=$2
start_tile=$3
end_tile=$4

# Extract and parse information from MAF file into tile_information
file="/stanley/WangLab/atlas/$tissue/01.data/$maf"
cat $file |sed 's/></>\n</g'|grep 'StageXPos'|cut -d ' ' -f 2-3,7-11|sed 's/"//g'|perl -pe 's/=.*? / /g'|sed 's/=.*$//g' |head -1 > head
cat $file |sed 's/></>\n</g'|grep 'StageXPos'|cut -d ' ' -f 2-3,7-11|sed 's/"//g'|perl -pe 's/[a-zA-Z].*?=//g' > a3
cat head a3|sed 's/ /	/g' > /stanley/WangLab/atlas/$tissue/01.data/tile_information
rm head a3

# Use tile_information to create orderlist 
python3 /stanley/WangLab/atlas/code/0.setup/get_maf_positions.py $tissue $sample $start_tile $end_tile




