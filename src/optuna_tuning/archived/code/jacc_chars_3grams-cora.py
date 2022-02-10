import os
import pandas as pd
import numpy as np
import time
import optuna
import sys, os
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import *
from datasets.common.create_dataset import createDataset, createTrueLabels
from datetime import datetime


# --- Dataset injection CORA --- #
CORA_groundTruth = os.path.abspath("../../data/coraIdDuplicates.csv")
CORA = os.path.abspath("../../data/coraProfiles.csv")
CORA_groundTruth = pd.read_csv(CORA_groundTruth,sep='|',header=None,names=['id1','id2'])
CORA_groundTruth=CORA_groundTruth.sort_values(by=['id1','id2'],ignore_index=True)
CORA = pd.read_csv(CORA,sep='|')
DATASET_NAME = 'CORA'
fields = ['address', 'author', 'editor', 'institution', 'month', 'note', 'pages','publisher', 'title', 'venue', 'volume', 'year']
data, true_matrix = createDataset(CORA, CORA_groundTruth, fields, 'Entity Id')
labels_groundTruth, numOfObjWithoutDups, groups = createTrueLabels(CORA['Entity Id'].tolist(),CORA_groundTruth)            
data_length = [ len(x) for x in data ]


# --- Results DF --- #
headers = ['trial_id','maxNumberOfClusters','maxDissimilarityDistance','similarityThreshold',
             'windowSize','metric','similarityVectors',"distanceMetricEmbedding","distanceMetric",
             "numberOfPermutations","ngramms","jaccardWithChars", 
             "numOfComparisons", "diffObjectsComparedSuccess", "sameObjectsCompared", "sameObjectsComparedSuccess",
             "selectionVariance","selectedNumOfPrototypes","averageBucketSize", "prototypesTime", "embeddingsTime", "wtaTime", "similarityTime",
             'Accuracy','Precision','Recall','F1','Time']
results_dataframe = pd.DataFrame(columns = headers)

SPEED_UP_RATE = 9

title = "jacc_chars_3grams-cora"
db_name = "WinnER_Experiments_v1"
storage_name = "sqlite:///{}.db".format(db_name)
study_name = title  # Unique identifier of the study.

if not os.path.exists(title + ".csv"):
    with open(title + ".csv", 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.close()
'''
 OPTUNA objective function
'''
def objective(trial):
    
    ngramms = trial.suggest_categorical("ngramms",[3]) 
    jaccard_withchars = trial.suggest_categorical("jaccard_withchars", [True])
    max_numberOf_clusters = trial.suggest_int("max_numberOf_clusters", 700, 1000) 
    distanceMetric = trial.suggest_categorical("distanceMetric", ["jaccard"])
    max_dissimilarityDistance= trial.suggest_float("max_dissimilarityDistance", 0.3, 0.8)
    prototypesFilterThr = trial.suggest_float("prototypesFilterThr", 0.1, 0.5)

    # --- Embedding phase
    distanceMetricEmbedding = trial.suggest_categorical("distanceMetricEmbedding", ["jaccard", "l_inf"])
    
    # -- WTA algorithm
    windowSize= trial.suggest_int("windowSize", 25, 150) 
    number_of_permutations= trial.suggest_int("number_of_permutations", 1, 5) 
    
    # -- Similarity evaluation
    similarityVectors = trial.suggest_categorical("similarityVectors", ["initial"])
    similarityThreshold = trial.suggest_float("similarityThreshold", 0.6, 0.75)
    metric = trial.suggest_categorical("metric",["kendal"]) 
    
    start = time.time()
    model = WinnER(
        max_numberOf_clusters= max_numberOf_clusters,
        max_dissimilarityDistance= max_dissimilarityDistance,
        windowSize= windowSize,
        similarityThreshold= similarityThreshold,
        metric=metric,
        similarityVectors=similarityVectors,
        number_of_permutations = number_of_permutations,
        distanceMetric= distanceMetric,
        distanceMetricEmbedding = distanceMetricEmbedding,
        ngramms= ngramms,
        jaccard_withchars = jaccard_withchars,
        prototypesFilterThr = prototypesFilterThr,
        verboseLevel = 0,
        maxNumberOfComparisons = int((CORA.shape[0]*CORA.shape[0])/SPEED_UP_RATE)
        #         disableTqdm = True
    )
    model.hackForDebug(labels_groundTruth, true_matrix)
    model = model.fit(data)
    if model == None:
        precision = 0.0
        f1 = 0.0
        recall = 0.0
    else:
        acc,f1,precision,recall = model.evaluate(model.mapping_matrix, true_matrix, with_confusion_matrix=False)

    exec_time = time.time() - start
    trial_id = trial.number
    row = [
        trial_id, max_numberOf_clusters,max_dissimilarityDistance,similarityThreshold,
        windowSize,metric,similarityVectors,distanceMetricEmbedding,distanceMetric,
        number_of_permutations,ngramms,jaccard_withchars, 
        model.numOfComparisons, model.diffObjectsComparedSuccess, model.sameObjectsCompared, model.sameObjectsComparedSuccess,
        model.selectionVariance,model.selected_numOfPrototypes,len(model.buckets.keys()),
        model.prototypes_time, model.embeddings_time, model.wta_time, model.similarity_time, acc, precision, recall, f1, exec_time]
    results_dataframe.loc[len(results_dataframe)+1] = row
    with open(title + ".csv", 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

    return recall

storage_name = "sqlite:///{}.db".format(db_name)
study = optuna.create_study(directions=["maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=50, show_progress_bar=True)
results_dataframe.to_pickle(study_name + datetime.now().strftime("_%m%d%H%M") + ".pkl")
f.close()