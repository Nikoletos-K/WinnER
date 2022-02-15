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

# --- Optuna script on CORA --- #
DISSIMILARITY_DISTANCE = None; CHAR_TOKENIZATION = None; NGRAMS = None;
if "-m" in set(sys.argv):
    METRIC_ARG = sys.argv[sys.argv.index("-m") + 1]
    DISSIMILARITY_DISTANCE = "jaccard" if sys.argv[sys.argv.index("-m") + 1] == "j" else "euclid_jaccard"
if "-t" in set(sys.argv):
    CHAR_TOKENIZATION = True if sys.argv[sys.argv.index("-t") + 1] == "1" else False 
    CHAR_LEVEL = "char"  if CHAR_TOKENIZATION else "word"
if "-ngrams" in set(sys.argv):
    NGRAMS = 2 if sys.argv[sys.argv.index("-ngrams") + 1] == "2" else 3
if "-ntrials" in set(sys.argv):
    num_of_trials = int(sys.argv[sys.argv.index("-ntrials") + 1])
else:
    num_of_trials = 50

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
headers = ['trial_id','max_num_of_clusters','max_dissimilarity_distance','similarity_threshold',
             'window_size','metric','similarity_vectors',"embedding_distance_metric","distance_metric",
             "number_of_permutations","ngrams","char_tokenization", 
             "num_of_comparisons", "diffObjectsComparedSuccess", "sameObjectsCompared", "sameObjectsComparedSuccess",
             "selection_variance","selectedNumOfPrototypes","averageBucketSize", "prototypesTime", "embeddingsTime", "wtaTime", "similarityTime",
             'Accuracy','Precision','Recall','F1','Time']

results_dataframe = pd.DataFrame(columns = headers)

SPEED_UP_RATE = 9

# title = "jacc_chars_3grams-cora"
title = DISSIMILARITY_DISTANCE + "_" + CHAR_LEVEL + "_" + str(NGRAMS) + "_" + "CORA"
print("EXPERIMENTS: ", title)
db_name = "WinnER_Experiments_v2"
storage_name = "sqlite:///{}.db".format(db_name)
study_name = title  # Unique identifier of the study.
db_path = "db"
csv_path = "csv"
df_path = "df"
cora_path = "cora"

if not os.path.exists(csv_path + "/" + title + ".csv"):
    with open("csv/" + title + ".csv", 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.close()

'''
 OPTUNA objective function
'''
def objective(trial):
    
    ngrams = NGRAMS
    char_tokenization = CHAR_TOKENIZATION
    distance_metric = DISSIMILARITY_DISTANCE
    max_num_of_clusters = None
    max_dissimilarity_distance = None
    embedding_distance_metric = DISSIMILARITY_DISTANCE
    # embedding_distance_metric = trial.suggest_categorical("embedding_distance_metric", [DISSIMILARITY_DISTANCE, "l_inf"])
    
    if CHAR_TOKENIZATION == True:
        max_num_of_clusters = trial.suggest_int("max_num_of_clusters", 200, 900) 
        max_dissimilarity_distance = trial.suggest_float("max_dissimilarity_distance", 0.1, 0.9)
    else:
        max_num_of_clusters = trial.suggest_int("max_num_of_clusters", 100, 900) 
        max_dissimilarity_distance = trial.suggest_float("max_dissimilarity_distance", 0.1, 0.9)

    
    # -- WTA algorithm
    window_size = trial.suggest_int("window_size", 30, 100) 
    number_of_permutations= trial.suggest_int("number_of_permutations", 1, 5) 
    
    # -- Similarity evaluation
    # similarity_vectors = trial.suggest_categorical("similarity_vectors", ["initial"])
    similarity_vectors = "initial"
    similarity_threshold = trial.suggest_float("similarity_threshold", 0.55, 0.75)
    # metric = trial.suggest_categorical("metric",["kendal", "spearman"]) 
    metric = "kendal"
    
    start = time.time()
    model = WinnER(
        max_num_of_clusters = max_num_of_clusters,
        max_dissimilarity_distance = max_dissimilarity_distance,
        window_size = window_size,
        similarity_threshold = similarity_threshold,
        metric = metric,
        similarity_vectors = similarity_vectors,
        number_of_permutations = number_of_permutations,
        distance_metric = distance_metric,
        embedding_distance_metric = embedding_distance_metric,
        ngrams = ngrams,
        char_tokenization = char_tokenization,
        verbose_level = -1,
        disable_tqdm = True
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
        trial_id, max_num_of_clusters,max_dissimilarity_distance,similarity_threshold,
        window_size,metric,similarity_vectors,embedding_distance_metric,distance_metric,
        number_of_permutations,ngrams,char_tokenization, 
        model.num_of_comparisons, model.diffObjectsComparedSuccess, model.sameObjectsCompared, model.sameObjectsComparedSuccess,
        model.selection_variance,model.selected_numOfPrototypes,len(model.buckets.keys()),
        model.prototypes_time, model.embeddings_time, model.wta_time, model.similarity_time, acc, precision, recall, f1, exec_time]
    results_dataframe.loc[len(results_dataframe)+1] = row
    with open(csv_path + "/" + title + ".csv", 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

    return recall

study_name = title  # Unique identifier of the study.
study = optuna.create_study(directions=["maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=num_of_trials, show_progress_bar=True)
results_dataframe.to_pickle(df_path + "/" + study_name + datetime.now().strftime("_%m%d%H%M") + ".pkl")
