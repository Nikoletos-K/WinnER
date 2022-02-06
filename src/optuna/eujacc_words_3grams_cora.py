import os
import pandas as pd
import numpy as np
import time
import optuna
import sys, os
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
results_dataframe = pd.DataFrame(
    columns=['maxNumberOfClusters','maxDissimilarityDistance','similarityThreshold',
             'windowSize','metric','similarityVectors',"distanceMetricEmbedding","distanceMetric",
             "numberOfPermutations","ngramms","jaccardWithChars", 
             "numOfComparisons", "diffObjectsComparedSuccess", "sameObjectsCompared", "sameObjectsComparedSuccess",
             "selectionVariance","selectedNumOfPrototypes","averageBucketSize", "prototypesTime", "embeddingsTime", "wtaTime", "similarityTime",
             'Accuracy','Precision','Recall','F1','Time']
)

SPEED_UP_RATE = 9


'''
 OPTUNA objective function
'''
def objective(trial):
    
    ngramms = trial.suggest_categorical("ngramms",[3]) 
    jaccard_withchars = trial.suggest_categorical("jaccard_withchars", [False])
    max_numberOf_clusters = trial.suggest_int("max_numberOf_clusters", 700, 1000) 
    distanceMetric = trial.suggest_categorical("distanceMetric", ["euclid_jaccard"])
    max_dissimilarityDistance= trial.suggest_float("max_dissimilarityDistance", 0.3, 0.8)
    prototypesFilterThr = trial.suggest_float("prototypesFilterThr", max_dissimilarityDistance-max_dissimilarityDistance/2, max_dissimilarityDistance-max_dissimilarityDistance/5)

    # --- Embedding phase
    distanceMetricEmbedding = trial.suggest_categorical("distanceMetricEmbedding", ["euclid_jaccard", "l_inf"])
    
    # -- WTA algorithm
    windowSize= trial.suggest_int("windowSize", 25, 150) 
    number_of_permutations= trial.suggest_int("number_of_permutations", 1, 5) 
    
    # -- Similarity evaluation
    similarityVectors = trial.suggest_categorical("similarityVectors", ["initial"])
    similarityThreshold = trial.suggest_float("similarityThreshold", 0.6, 0.75)
    metric = trial.suggest_categorical("metric",["kendal"]) 
    
    start = time.time()
    try:
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
        results_dataframe.loc[len(results_dataframe)+1] = [
            max_numberOf_clusters,max_dissimilarityDistance,similarityThreshold,
            windowSize,metric,similarityVectors,distanceMetricEmbedding,distanceMetric,
            number_of_permutations,ngramms,jaccard_withchars, 
            model.numOfComparisons, model.diffObjectsComparedSuccess, model.sameObjectsCompared, model.sameObjectsComparedSuccess,
            model.selectionVariance,model.selected_numOfPrototypes,len(model.buckets.keys()),
            model.prototypes_time, model.embeddings_time, model.wta_time, model.similarity_time, acc, precision, recall, f1, exec_time]
    except Warning as e:
        print(e)
        # trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return recall

title = "eujacc_words_3gramms-cora"
study_name = title  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(title)
study = optuna.create_study(directions=["maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=50, show_progress_bar=True)
results_dataframe.to_pickle(study_name + datetime.now().strftime("_%m%d%H%M") + ".pkl")
