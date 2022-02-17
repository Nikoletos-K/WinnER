from tqdm import tqdm as tqdm
import math
import numpy as np
import networkx as nx

from ..utils.text_process import preprocess

def createDataset(cora_dataframe, true_values, fields, id_column, keepNone = False, preprocessEnabled=True):

    rawStr_col = []
    index_to_id_dict = {}
    sameEntities_dictionary = {}

    i=0
    for _, row in tqdm(cora_dataframe.iterrows(), total = len(cora_dataframe), desc="Creating dataset from imput files", dynamic_ncols = True):
        index_to_id_dict[int(row[id_column])] = i
        rawStr = []
        for field in fields:
            if (isna(row[field]) and keepNone == True) or (keepNone == False and not isna(row[field])):
                rawStr.append(str(row[field]))
        i+=1
        if preprocessEnabled:
            rawStr_col.append(preprocess(rawStr))
        else:
            rawStr_col.append(rawStr)
            
    num_of_records = len(cora_dataframe)
    trueValues_matrix = np.zeros([num_of_records,num_of_records],dtype=np.int8)

    for _, row in tqdm(true_values.iterrows(), total = len(true_values), desc="Creating groundtruth matrix", dynamic_ncols = True):  
        trueValues_matrix[index_to_id_dict[row['id1']]][index_to_id_dict[row['id2']]] = 1
        trueValues_matrix[index_to_id_dict[row['id2']]][index_to_id_dict[row['id1']]] = 1

    return rawStr_col, trueValues_matrix

def createTrueLabels(idColumn,groundTruth):

    data = list(zip(groundTruth.id1, groundTruth.id2))
    G = nx.Graph()
    G.add_edges_from(data)
    groups = list(nx.connected_components(G))
    newId = len(groups)
    labels_groundTruth = np.empty([len(idColumn)], dtype=int)
    for tid in idColumn:
        for g,g_index in zip(groups,range(0,len(groups),1)):
            if tid in g:
                labels_groundTruth[tid] = g_index

        if labels_groundTruth[tid] not in range(0,len(groups),1):
            labels_groundTruth[tid] = newId
            newId+=1
            
    return labels_groundTruth,newId,groups


def isna(value):
    if isinstance(value, float) and math.isnan(value):
        return True 
    else:
        return False
    
