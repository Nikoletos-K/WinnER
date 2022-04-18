from importlib_metadata import files
import pandas as pd
import numpy as np
import time
import optuna
import sys, os
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import *
from datasets.common.create_dataset import process_dataset, create_true_labels
from datetime import datetime
import logging

# --- Optuna script on CORA --- #
DISSIMILARITY_DISTANCE = None; CHAR_TOKENIZATION = None; NGRAMS = None;

if "-d" in set(sys.argv):
    DATASET = sys.argv[sys.argv.index("-d") + 1]
else:
    logging.error("Please provide dataset (-dataset <dataset_name>)")
    sys.exit()
print("Dataset: " + DATASET)

if "-metric" in set(sys.argv):
    METRIC_ARG = sys.argv[sys.argv.index("-metric") + 1]
    DISSIMILARITY_DISTANCE = "jaccard" if sys.argv[sys.argv.index("-metric") + 1] == "j" else "euclid_jaccard"
else:
    METRIC_ARG =  DISSIMILARITY_DISTANCE = "euclid_jaccard"
print("Dissimilarity metric: " + DISSIMILARITY_DISTANCE)


if "-tok" in set(sys.argv):
    CHAR_TOKENIZATION = True if sys.argv[sys.argv.index("-tok") + 1] == "1" else False 
    TOKENIZATION_LEVEL = "char"  if CHAR_TOKENIZATION else "word"
else:
    CHAR_TOKENIZATION = True
    TOKENIZATION_LEVEL = "char"
print("Tokenization: " + TOKENIZATION_LEVEL)
if "-ngrams" in set(sys.argv):
    NGRAMS = 2 if sys.argv[sys.argv.index("-ngrams") + 1] == "2" else 3
else:
    NGRAMS = 3
print("N-grams: " + str(NGRAMS))
if "-ntrials" in set(sys.argv):
    num_of_trials = int(sys.argv[sys.argv.index("-ntrials") + 1])
else:
    num_of_trials = 50
print("Number of trials: " + str(num_of_trials))

# --- Dataset injection --- #
dataset_shuffled=None; dataset=None;  dataset_ground_truth=None;
MISSING_RATIO = 10
column_id = 'Entity Id'
if DATASET == "CORA":
    dataset_ground_truth = pd.read_csv(os.path.abspath("../../data/coraIdDuplicates.csv"), sep='|', header=None, names=['id1','id2'])
    dataset = pd.read_csv(os.path.abspath("../../data/coraProfiles.csv"), sep='|')
elif DATASET == "CENSUS":
    dataset_ground_truth = pd.read_csv(os.path.abspath("../../data/censusIdDuplicates.csv"), sep='|', header=None, names=['id1','id2'])
    dataset = pd.read_csv(os.path.abspath("../../data/censusProfiles.csv"), sep='|')
elif DATASET == "CDDB":
    dataset_ground_truth = pd.read_csv(os.path.abspath("../../data/cddbIdDuplicates.csv"), sep='/00000', engine='python', header=None,names=['id1','id2'])
    dataset = pd.read_csv(os.path.abspath("../../data/cddbProfiles.csv"), sep='/00000', engine='python')
elif DATASET == "SIGMOD2022_X1":
    Y1 = pd.read_csv(os.path.abspath("../../data/Y1.csv"), sep = ',')
    X1 = pd.read_csv(os.path.abspath("../../data/X1.csv"), sep = ',')
    id_to_index = {}
    new_id = []
    for r in X1.iterrows():
        id_to_index[r[1]['id']] = r[0]
        new_id.append(r[0])
    X1[column_id] = new_id
    new_id1 = []
    new_id2 = []
    for r in Y1.iterrows():
        new_id1.append(id_to_index[r[1]['id1']])
        new_id2.append(id_to_index[r[1]['id2']])
    Y1['id1'] = new_id1
    Y1['id2'] = new_id2
    dataset_ground_truth = Y1
    dataset = X1
elif DATASET == "SIGMOD2022_X2":
    Y2 = pd.read_csv(os.path.abspath("../../data/Y2.csv"), sep = ',')
    X2 = pd.read_csv(os.path.abspath("../../data/X2.csv"), sep = ',')
    id_to_index = {}
    new_id = []
    for r in X2.iterrows():
        id_to_index[r[1]['id']] = r[0]
        new_id.append(r[0])
    X2[column_id] = new_id
    new_id1 = []
    new_id2 = []
    for r in Y2.iterrows():
        new_id1.append(id_to_index[r[1]['id1']])
        new_id2.append(id_to_index[r[1]['id2']])
    Y2['id1'] = new_id1
    Y2['id2'] = new_id2
    dataset_ground_truth = Y2
    dataset = X2
else:
    logging.error("Available datasets: CORA, CDDB, CENSUS")
    sys.exit()

SPEED_UP_RATE = 9
if "-v" in set(sys.argv):
    version = sys.argv[sys.argv.index("-v") + 1]
    db_name = "WinnER_Experiments_v" + version
else:
    version = 2
    db_name = "WinnER_Experiments_v2"
print("Version: " + version)

title = DISSIMILARITY_DISTANCE + "_" + TOKENIZATION_LEVEL + "_" + str(NGRAMS)  + "_" + "v" + version + "_" + DATASET
storage_name = "sqlite:///{}.db".format(db_name)
study_name = title  # Unique identifier of the study.
print("Study name: " + study_name)
db_path = "db"
csv_path = "csv"
df_path = "df"


if os.path.exists(df_path + "/" + study_name + "DF" + ".pkl"):
    dataset_shuffled = pd.read_pickle(df_path + "/" + study_name + "DF" + ".pkl")  
    print("Using an existing shuffled data set: " + df_path + "/" + study_name + "DF" + ".pkl")
else:
    dataset_shuffled = dataset.sample(frac=1).reset_index(drop=True)
    print("Shuffled data set")


# --- Columns based on NAN values --- #
na_df = (dataset.isnull().sum() / len(dataset)) * 100
na_df = na_df.sort_values(ascending=False)
missing_data = pd.DataFrame({'missing_ratio' : na_df})
fields = missing_data[missing_data['missing_ratio'] < MISSING_RATIO].index.values.tolist()
fields.remove(column_id)
print("Dataset attributes: " + ", ".join(fields))
data, true_matrix = process_dataset(dataset_shuffled, dataset_ground_truth, fields, column_id)
labels_groundTruth, numOfObjWithoutDups, groups = create_true_labels(dataset[column_id].tolist(),dataset_ground_truth)            

# --- Results DF --- #
headers = ['trial_id','max_num_of_clusters','max_dissimilarity_distance','similarity_threshold',
             'window_size','metric','similarity_vectors',"embedding_distance_metric","distance_metric",
             "number_of_permutations","ngrams","char_tokenization", 
             "num_of_comparisons", "diffObjectsComparedSuccess", "sameObjectsCompared", "sameObjectsComparedSuccess",
             "selection_variance","selectedNumOfPrototypes","averageBucketSize", "prototypesTime", "embeddingsTime", "wtaTime", "similarityTime",
             'Accuracy','Precision','Recall','F1','Time']

results_dataframe = pd.DataFrame(columns = headers)
dataset_shuffled.to_pickle(df_path + "/" + study_name + "DF" + ".pkl")

if not os.path.exists(csv_path + "/" + title + ".csv"):
    with open("csv/" + title + ".csv", 'w+') as f:
        print("Data will be stored in: " + title + ".csv")
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
        max_num_of_clusters = trial.suggest_int("max_num_of_clusters", 100, int(dataset.shape[0]/2)) 
        max_dissimilarity_distance = trial.suggest_float("max_dissimilarity_distance", 0.05, 0.95)
    else:
        max_num_of_clusters = trial.suggest_int("max_num_of_clusters", 100, int(dataset.shape[0]/2)) 
        max_dissimilarity_distance = trial.suggest_float("max_dissimilarity_distance", 0.05, 0.95)

    
    # -- WTA algorithm
    window_size = trial.suggest_int("window_size", 5, 100) 
    number_of_permutations= trial.suggest_int("number_of_permutations", 1, 5) 
    
    # -- Similarity evaluation
    # similarity_vectors = trial.suggest_categorical("similarity_vectors", ["initial"])
    similarity_vectors = "initial"
    similarity_threshold = trial.suggest_float("similarity_threshold", 0.45, 0.75)
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
print("Optuna trials starting")
study.optimize(objective, n_trials=num_of_trials, show_progress_bar=True)
print("Optuna trials finished")
results_dataframe.to_pickle(df_path + "/" + study_name + datetime.now().strftime("_%m%d%H%M") + ".pkl")
print("All results saved as dataframe to file " + study_name + datetime.now().strftime("_%m%d%H%M") + ".pkl")