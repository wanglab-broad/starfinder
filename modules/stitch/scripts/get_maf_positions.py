# Take extracted MAF information and format into a list of position numbers, with 0 representing a blank tile
# IMPORTANT: this script is employed by `extract_from_maf.sh` shell script, which generates the required tile_information file from the .maf
import pandas as pd
import numpy as np
import copy
import os
from scipy.sparse import coo_matrix
import argparse 

def get_positions(args):
    a = pd.read_csv(f"/stanley/WangLab/atlas/{args.tissue}/01.data/tile_information", sep='\t', header=0, index_col=None)
    a['Position_num'] = np.array(a.loc[:,'PositionIdentifier'].apply(lambda x: x.replace('Position','')))
    a_filter = copy.deepcopy(a.loc[(a.loc[:,'Position_num'].astype(int)<= args.end) & (a.loc[:,'Position_num'].astype(int) >= args.start),:])
    b = a['StageYPos'].shift(-1) - a['StageYPos']
    step = pd.value_counts(b).index.values[0]
    point_relative = [min(np.array(a_filter.loc[:,'StageXPos'])),min(np.array(a_filter.loc[:,'StageYPos']))]
    point_relative_max = [max(np.array(a_filter.loc[:,'StageXPos'])),max(np.array(a_filter.loc[:,'StageYPos']))]
    coo_shape = np.array(point_relative_max)-np.array(point_relative)
    coo_shape = (coo_shape / step + 0.5 + 1).astype(int)
    a_filter['relative_x'] = np.array(((a_filter['StageXPos'] - point_relative[0])/step + 0.5),dtype=int)
    a_filter['relative_y'] = np.array(((a_filter['StageYPos'] - point_relative[1])/step + 0.5),dtype=int)
    a_filter_2 = copy.deepcopy(a_filter.loc[:,['Position_num', 'relative_x', 'relative_y']])
    value = np.array(a_filter_2.loc[:,'Position_num'],dtype=int)
    row = np.array(a_filter_2.loc[:,'relative_y'],dtype=int)
    col = np.array(a_filter_2.loc[:,'relative_x'],dtype=int)
    matrix_shape = [coo_shape[1],coo_shape[0]]
    c = coo_matrix((value, (row,col)), shape=matrix_shape).toarray()
    c1 = pd.DataFrame(c)
    d = c1.unstack(1)
    d2 = pd.DataFrame(d)
    d2 = d2.reset_index()

    return d2

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("tissue", type=str, help="Name of sample")
parser.add_argument("sample", type=str, help="Name of sample within tissue")
parser.add_argument("start", type=int, help="Position number of first tile in sample")
parser.add_argument("end", type=int, help="Position number of last tile in sample")
args = parser.parse_args()

d = get_positions(args)
d.iloc[:,2].to_csv(f'/stanley/WangLab/atlas/{args.tissue}/00.sh/00.orderlist/{args.sample}_orderlist.csv',index= False,header = False)
d.to_csv(f'/stanley/WangLab/atlas/{args.tissue}/00.sh/00.orderlist/{args.sample}_orderlist_raw_df.csv',index= False,header = False)

