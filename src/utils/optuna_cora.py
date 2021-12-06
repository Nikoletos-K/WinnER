import pandas as pd
import numpy as np
import collections
import editdistance
import string
import sklearn
import pandas_read_xml as pdx
import time
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import math
import os
import scipy.special as special
import igraph
import networkx as nx
import itertools 
import sklearn
import plotly.express as px
import logging
import sys
import optuna

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from tqdm.notebook import tqdm as tqdm
from scipy.spatial.distance import directed_hausdorff,hamming
from scipy.stats._stats import _kendall_dis
from scipy.stats import spearmanr,kendalltau,pearsonr,kruskal,mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import jaro_similarity,jaro_winkler_similarity,jaccard_distance
from sklearn.metrics import jaccard_score,accuracy_score,auc,f1_score,recall_score,precision_score,classification_report
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy import stats 
from scipy.spatial.distance import euclidean,hamming,jaccard
from matplotlib.patches import Rectangle
from sklearn.metrics import ndcg_score
from datetime import datetime
from sklearn.decomposition import PCA 
from numpy.linalg import svd

# plt.style.use('seaborn-white') # plot style


from model import RankedWTAHash, customClassificationReport, report, set_params
from utils.classification import MMD2, myMMD2
from plot.heatmap import myHeatmap
from plot.confusion_matrix import create_ConfusionMatrix
from plot.dimension_reduction import SpaceVisualization2D, SpaceVisualization3D, SpaceVisualizationEmbeddings2D, SpaceVisualizationEmbeddings3D
from plot.heatmap import similarityProbsHeatMap, similarityProbsHeatMapWithClusters
from datasets.common.create_dataset import createDataset, createTrueLabels

CORA_groundTruth = os.path.abspath("../data/coraIdDuplicates.csv")
CORA = os.path.abspath("../data/coraProfiles.csv")
CORA_groundTruth = pd.read_csv(CORA_groundTruth,sep='|',header=None,names=['id1','id2'])
CORA_groundTruth=CORA_groundTruth.sort_values(by=['id1','id2'],ignore_index=True)
CORA = pd.read_csv(CORA,sep='|')
fields = [
     'address',
     'author',
     'editor',
     'institution',
     'month',
     'note',
     'pages',
     'publisher',
     'title',
     'venue',
     'volume',
     'year',
     'Unnamed: 13'
]
data, true_matrix = createDataset(CORA, CORA_groundTruth, fields, 'Entity Id')
labels_groundTruth, numOfObjWithoutDups, groups = createTrueLabels(CORA['Entity Id'].tolist(),CORA_groundTruth)            
data_length = [ len(x) for x in data ]
speedUpRate = 9
results_dataframe = pd.DataFrame(
    columns=['maxNumberOfClusters','maxDissimilarityDistance','similarityThreshold',
             'windowSize','metric','similarityVectors',"distanceMetricEmbedding","distanceMetric",
             "numberOfPermutations","ngramms","jaccardWithChars", 
             "numOfComparisons", "diffObjectsComparedSuccess", "sameObjectsCompared", "sameObjectsComparedSuccess",
             "selectionVariance","selectedNumOfPrototypes","averageBucketSize",
             'Accuracy','Precision','Recall','F1','Time']
)

def objective(trial):

    ngramms= trial.suggest_int("ngramms", 2, 3) 
    jaccard_withchars = trial.suggest_categorical("jaccard_withchars", [True, False])
    max_numberOf_clusters= trial.suggest_int("max_numberOf_clusters", 500, 1500) 
    distanceMetric = trial.suggest_categorical("distanceMetric", ["jaccard", "euclid_jaccard"])
    max_dissimilarityDistance= trial.suggest_float("max_dissimilarityDistance", 0.3, 0.8)
    prototypesFilterThr = trial.suggest_float("prototypesFilterThr", 0.5, 0.9)
    # --- Embedding phase
    distanceMetricEmbedding = trial.suggest_categorical("distanceMetricEmbedding", ["l_inf", "jaccard", "euclid_jaccard"])
    # -- WTA algorithm
    windowSize= trial.suggest_int("windowSize", 15, 80) 
    number_of_permutations= trial.suggest_int("number_of_permutations", 1, 5) 
    # -- Similarity evaluation
    similarityVectors = trial.suggest_categorical("similarityVectors", ["initial"])
    similarityThreshold = trial.suggest_float("similarityThreshold", 0.6, 0.8)
    metric = trial.suggest_categorical("metric", ["pearson", "spearman", "kendal", "rbo"])
    
    start = time.time()
    model = RankedWTAHash(
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
        maxNumberOfComparisons = int((CORA.shape[0]*CORA.shape[0])/speedUpRate)
#         disableTqdm = True
    )
    model.hackForDebug(labels_groundTruth, true_matrix)
    model = model.fit(data)
    if model == None:
        precision = 0.0
        f1 = 0.0
        recall = 0.0
    else:
        acc,f1,precision,recall = model.evaluate(model.mapping_matrix, true_matrix)
    exec_time = time.time() - start
    results_dataframe.loc[len(results_dataframe)+1] = [
        max_numberOf_clusters,max_dissimilarityDistance,similarityThreshold,
        windowSize,metric,similarityVectors,distanceMetricEmbedding,distanceMetric,
        number_of_permutations,ngramms,jaccard_withchars, 
        model.numOfComparisons, model.diffObjectsComparedSuccess, model.sameObjectsCompared, model.sameObjectsComparedSuccess,
        model.selectionVariance,model.selected_numOfPrototypes,len(model.buckets.keys()),acc,precision,recall,f1,exec_time]

    return recall

import datetime
x = datetime.datetime.now()
study_name = "best-params-jaccard-recall-" + x.strftime("%d-%m_%H:%M")  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(directions=["maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)

print("Number of finished trials: ", len(study.trials))

study.optimize(objective, n_trials=30, show_progress_bar=True)
results_dataframe.to_pickle(study_name+".pkl")