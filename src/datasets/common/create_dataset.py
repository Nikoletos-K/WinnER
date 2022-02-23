from tqdm import tqdm as tqdm
import math
import numpy as np
import networkx as nx

from ..utils.text_process import preprocess

def process_dataset(dataset, true_values, fields, column_id, keep_none = False, enable_preprocess = True):

    raw_str_col = []
    index_to_id_dict = {}

    i=0
    for _, row in tqdm(dataset.iterrows(), total = len(dataset), desc="Creating dataset from imput files", dynamic_ncols = True):
        index_to_id_dict[int(row[column_id])] = i
        raw_str = []
        for field in fields:
            if (isna(row[field]) and keep_none == True) or (keep_none == False and not isna(row[field])):
                raw_str.append(str(row[field]))
        i+=1
        if enable_preprocess:
            raw_str_col.append(preprocess(raw_str))
        else:
            raw_str_col.append(raw_str)
            
    num_of_records = len(dataset)
    true_values_matrix = np.zeros([num_of_records,num_of_records],dtype=np.int8)

    for _, row in tqdm(true_values.iterrows(), total = len(true_values), desc="Creating groundtruth matrix", dynamic_ncols = True):  
        true_values_matrix[index_to_id_dict[row['id1']]][index_to_id_dict[row['id2']]] = 1
        true_values_matrix[index_to_id_dict[row['id2']]][index_to_id_dict[row['id1']]] = 1

    return raw_str_col, true_values_matrix

def create_true_labels(column_id, ground_truth_values):

    data = list(zip(ground_truth_values.id1, ground_truth_values.id2))
    G = nx.Graph()
    G.add_edges_from(data)
    groups = list(nx.connected_components(G))
    newId = len(groups)
    labels_ground_truth = np.empty([len(column_id)], dtype=int)
    for tid in column_id:
        for g,g_index in zip(groups,range(0,len(groups),1)):
            if tid in g:
                labels_ground_truth[tid] = g_index

        if labels_ground_truth[tid] not in range(0,len(groups),1):
            labels_ground_truth[tid] = newId
            newId+=1
            
    return labels_ground_truth, newId, groups


def isna(value):
    if isinstance(value, float) and math.isnan(value):
        return True 
    else:
        return False
    
