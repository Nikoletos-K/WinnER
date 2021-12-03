'''
@author: Konstantinos Nikoletos, 2021
'''
import numpy as np
import editdistance
import sklearn
import time
import warnings
import nltk
import math

from tqdm.notebook import tqdm as tqdm
from scipy.spatial.distance import hamming
from scipy.stats._stats import _kendall_dis
from scipy.stats import spearmanr,kendalltau,pearsonr,kruskal,mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import jaccard_distance
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, recall_score, precision_score, classification_report
from scipy import stats 
from scipy.spatial.distance import hamming,jaccard
from sklearn.metrics import ndcg_score


# --------------------------------- #
# ---- Import from local files ---- #
# --------------------------------- #

from hash.wta import WTA, wtaSimilarity
from plot.dimension_reduction import SpaceVisualization2D, SpaceVisualization3D, SpaceVisualizationEmbeddings3D
from plot.heatmap import myHeatmap
from plot.confusion_matrix import create_ConfusionMatrix
from utils.metrics import *

# --------------------------------- #
# ----    Main model class     ---- #
# --------------------------------- #

class RankedWTAHash:

    def __init__(self, max_numberOf_clusters, max_dissimilarityDistance, windowSize, 
                 number_of_permutations=1, min_numOfNodes = 2, jaccard_withchars =True,
                 distanceMetricEmbedding = 'euclidean', metric = 'kendal', similarityVectors='ranked', 
                 distanceMetric = 'edit', prototypesFilterThr = None, ngramms = None, 
                 similarityThreshold = None, maxOnly = None, earlyStop=0, 
                 verboseLevel=0, rbo_p = 0.7, wtaM = 1, disableTqdm = False):
        '''
          Constructor
        '''
        self.max_numberOf_clusters = max_numberOf_clusters
        self.pairDictionary = dict()
        self.max_dissimilarityDistance = max_dissimilarityDistance
        self.windowSize = windowSize
        self.S_set = None
        self.S_index = None
        self.similarityThreshold = similarityThreshold
        self.maxOnly = maxOnly
        self.metric = metric
        self.min_numOfNodes = min_numOfNodes
        self.similarityVectors = similarityVectors
        self.number_of_permutations = number_of_permutations
        self.distanceMetric = distanceMetric
        self.distanceMetricEmbedding = distanceMetricEmbedding
        self.ngramms = ngramms
        self.jaccard_withchars =  jaccard_withchars
        self.prototypesFilterThr = prototypesFilterThr
        self.earlyStop = earlyStop
        self.selectionVariance = None
        self.numOfComparisons = 0
        self.verboseLevel = verboseLevel
        self.rbo_p = rbo_p
        self.wtaM = wtaM
        self.disableTqdm = disableTqdm

    def hackForDebug(self, labels_groundTruth, true_matrix):
        self.labels_groundTruth = labels_groundTruth
        self.true_matrix = true_matrix


    def fit(self, X):
        """
          Fit the classifier from the training dataset.
          Parameters
          ----------
          X : Training data.
          
          Returns
          -------
          self : The fitted classifier.
        """

        if self.verboseLevel >=0 :
            print("\n#####################################################################\n#     .~ RankedWTAHash with Vantage embeddings starts training ~.   #\n#####################################################################\n")

        if isinstance(X, list):
            input_strings = X
        else:
            input_strings = list(X)

        self.initialS_set = np.array(input_strings,dtype=object)
        self.S_set = np.array(input_strings,dtype=object)
        if self.distanceMetric == 'jaccard' and self.jaccard_withchars == False:
            for i in range(0,len(input_strings)):
                self.S_set[i] = set(nltk.ngrams(nltk.word_tokenize(self.S_set[i]), n=self.ngramms))
        elif self.distanceMetric == 'jaccard' and self.jaccard_withchars == True:
            for i in range(0,len(input_strings)):
                self.S_set[i] = set(nltk.ngrams(self.S_set[i], n=self.ngramms))

        self.S_index = np.arange(0,len(input_strings),1)
        
        if self.verboseLevel > 1:
            print("\n\nString positions are:")
            print(self.S_index)
            print("\n")

        if self.verboseLevel >=0 :
            print("###########################################################\n# > 1. Prototype selection phase                          #\n###########################################################\n")
            print("\n-> Finding prototypes and representatives of each cluster:")
        
        prototypes_time = time.time()
        self.prototypeArray,self.selected_numOfPrototypes = self.Clustering_Prototypes(self.S_index,self.max_numberOf_clusters, self.max_dissimilarityDistance, self.pairDictionary)
        self.embeddingDim = self.prototypeArray.size
        
        if self.verboseLevel > 0:
            print("\n- Prototypes selected:")
            print(self.prototypeArray)
            heatmapData = []
            for pr in self.prototypeArray:
                print(pr," -> ",self.initialS_set[pr])
                heatmapData.append(self.S_set[pr])            
            if self.selected_numOfPrototypes > 2:
                self.selectionVariance = myHeatmap(self.prototypeArray,self.metric,self.dissimilarityDistance)
                print("\n- Mean variance in prototype selection: ", self.selectionVariance)

        prototypes_time = time.time() - prototypes_time
        if self.verboseLevel >=0 :
            print("\n- Final number of prototypes: ",self.selected_numOfPrototypes )
            print("\n# Finished in %.6s secs" % (prototypes_time))
            print("\n")

        if self.earlyStop==1:
            return self

        if self.verboseLevel >=0 :
            print("###########################################################\n# > 2. Embeddings based on the Vantage objects            #\n###########################################################\n")
            print("\n-> Creating Embeddings:")
        embeddings_time = time.time()
        self.Embeddings = self.CreateVantageEmbeddings(self.S_index, self.prototypeArray, self.pairDictionary)
   
        if self.verboseLevel >=0 :
            print("- Embeddings created")
       
        if self.verboseLevel > 0:
            print(self.Embeddings)
            SpaceVisualization2D(self.Embeddings, self.prototypeArray)        
        
        embeddings_time = time.time() - embeddings_time

        if self.verboseLevel >=0 :
            print("\n# Finished in %.6s secs" % (embeddings_time))
            print("\n")

        if self.earlyStop==2:
            return self

        if self.verboseLevel >=0 :
            print("###########################################################\n# > 3. WTA Hashing                                        #\n###########################################################\n")
            print("\n-> Creating WTA Buckets:")

        wta_time = time.time()
        wta = WTA(self.windowSize, self.number_of_permutations, self.wtaM, self.disableTqdm)
        self.HashedClusters, self.buckets, self.rankedVectors = wta.fit(self.Embeddings)
        
        if self.verboseLevel > 0:
            print("- WTA buckets: ")
            for key in self.buckets.keys():
                print(key," -> ",self.buckets[key])
        
        if self.verboseLevel >=0 :
            print("\n- WTA number of buckets: ", len(self.buckets.keys()))
        
        if self.verboseLevel > 1:
            print("\n- WTA RankedVectors after permutation:")
            print(self.rankedVectors)

        if self.verboseLevel > 0:
            if self.similarityVectors == 'ranked':
                SpaceVisualizationEmbeddings3D(self.rankedVectors, self.prototypeArray, self.HashedClusters, withgroundruth=True, groundruth = self.labels_groundTruth, title='PCA visualization GroundTruth')
            elif self.similarityVectors == 'initial':
                SpaceVisualizationEmbeddings3D(self.Embeddings, self.prototypeArray, self.HashedClusters, withgroundruth=True, groundruth = self.labels_groundTruth, title='PCA visualization GroundTruth')

        wta_time = time.time() - wta_time

        if self.verboseLevel >=0 :
            print("\n# Finished in %.6s secs" % (wta_time))
            print("\n")
        
        if self.earlyStop==3:
            return self

        if self.verboseLevel >=0 :
            print("###########################################################\n# > 4. Similarity checking                                #\n###########################################################\n")
            print("\n-> Similarity checking:")

        similarity_time = time.time()

        if self.similarityVectors == 'ranked':
            self.mapping, self.mapping_matrix = self.SimilarityEvaluation(self.buckets,self.rankedVectors,self.similarityThreshold,maxOnly=self.maxOnly, metric=self.metric)
        elif self.similarityVectors == 'initial':
            self.mapping, self.mapping_matrix = self.SimilarityEvaluation(self.buckets,self.Embeddings,self.similarityThreshold,maxOnly=self.maxOnly, metric=self.metric)
        else:
            warnings.warn("similarityVectors: Available options are: ranked,initial")
        
        if self.mapping == None and self.mapping_matrix == None:
            return None

        if self.verboseLevel > 1:
            print("- Similarity mapping in a matrix")
            print(self.mapping_matrix)
        
        if self.verboseLevel > 0:
            print("\n- Total number of comparisons made: ", self.numOfComparisons)
            print("\n- Total number of comparisons of same objects: ", self.sameObjectsCompared)
            print("\n- Total number of comparisons of same objects with success: ", self.sameObjectsComparedSuccess)
            print("\n- Total number of comparisons of different objects with success: ", self.diffObjectsComparedSuccess)
        
        similarity_time = time.time() - similarity_time

        if self.verboseLevel >=0 :
            print("\n# Finished in %.6s secs" % (similarity_time))
            print("\n#####################################################################\n#                           .~  End  ~.                             #\n#####################################################################\n")

        return self

    def dissimilarityDistance(self, str1,str2,verbose=False):
        if self.verboseLevel > 2:
            print("-> ", self.initialS_set[str1])
            print("--> ", self.initialS_set[str2])

        if ((str1,str2) or (str2,str1))  in self.pairDictionary.keys():
            return self.pairDictionary[(str1,str2)]
        else:
            if self.distanceMetric == 'edit':
                distance = editdistance.eval(self.S_set[str1],self.S_set[str2])
            elif self.distanceMetric == 'jaccard':
                distance = jaccard_distance(self.S_set[str1],self.S_set[str2])
            else:
                warnings.warn("Available metrics for space creation: edit, jaccard ")
            self.pairDictionary[(str2,str1)] = self.pairDictionary[(str1,str2)] = distance
            
            if self.verboseLevel > 2:
                print(distance)
            
            return distance

    #####################################################################
    # 1. Prototype selection algorithm                                  #
    #####################################################################

    '''
    Clustering_Prototypes(S,k,d,r,C) 
    The String Clustering and Prototype Selection Algorithm
    is the main clustering method, that takes as input the intial strings S, 
    the max number of clusters to be generated in k,
    the maximum allowable distance of a string to join a cluster in var d
    and returns the prototype for each cluster in array Prototype
    '''
    def Clustering_Prototypes(self,S,k,d,pairDictionary,verbose=False):

        # ----------------- Initialization phase ----------------- #
        i = 0
        j = 0
        C = np.empty([S.size], dtype=int)
        r = np.empty([2,k],dtype=object)

        Clusters = [ [] for l in range(0,k)]

        for i in tqdm(range(0,S.size,1), desc="Prototype selection", disable = self.disableTqdm, dynamic_ncols = True):     # String-clustering phase, for all strings
            while j < k :       # iteration through clusters, for all clusters
                if r[0][j] == None:      # case empty first representative for cluster j
                    r[0][j] = S[i]   # init cluster representative with string i
                    C[i] = j         # store in C that i-string belongs to cluster j
                    Clusters[j].append(S[i])
                    break
                elif r[1][j] == None and (self.dissimilarityDistance(S[i],r[0][j]) <= d):  # case empty second representative
                    r[1][j] = S[i]                                             # and ED of representative 1  smaller than i-th string
                    C[i] = j
                    Clusters[j].append(S[i])
                    break
                elif (r[0][j] != None and r[1][j] != None) and (self.dissimilarityDistance(S[i],r[0][j]) + self.dissimilarityDistance(S[i],r[1][j])) <= d:
                    C[i] = j
                    Clusters[j].append(S[i])
                    break
                else:
                    j += 1
            i += 1

        # ----------------- Prototype selection phase ----------------- #

        Projections = np.empty([k],dtype=object)
        Prototypes = np.empty([k],dtype=int)
        sortedProjections = np.empty([k],dtype=object)
        Projections = []
        Prototypes = []
        sortedProjections = []

        if self.verboseLevel > 2:
            print("- - - - - - - - -")
            print("Cluster array:")
            print(C)
            print("- - - - - - - - -")
            print("Represantatives array:")
            print(r)
            print("- - - - - - - - -")
            print("Clusters:")
            print(Clusters)
            print("- - - - - - - - -")
            print("k:")
            print(k)
            print("- - - - - - - - -")

        new_numofClusters = k
        prototype_index = 0
        for j in range(0,k,1):
            
            apprxDistances = self.ApproximatedProjectionDistancesofCluster(r[1][j], r[0][j], j, Clusters[j], pairDictionary)
            
            if apprxDistances == None:
                new_numofClusters-=1
                continue
            
            Projections.append(apprxDistances)
            sortedProjections.append({new_numofClusters: v for new_numofClusters, v in sorted(Projections[prototype_index].items(), key=lambda item: item[1])})
            Prototypes.append(self.median(sortedProjections[prototype_index]))
            prototype_index += 1
        
        Prototypes, new_numofClusters = self.OptimizeClusterSelection(Prototypes, new_numofClusters)

        
        return np.array(Prototypes), new_numofClusters


    def ApproximatedProjectionDistancesofCluster(self, right_rep, left_rep, cluster_id, clusterSet, pairDictionary):

        distances_vector = dict()

        if len(clusterSet) > 2:
            rep_distance = self.dissimilarityDistance(right_rep,left_rep)

            for str_inCluster in range(0, len(clusterSet)):
                if clusterSet[str_inCluster] != right_rep and clusterSet[str_inCluster] != left_rep:
                    right_rep_distance = self.dissimilarityDistance(right_rep,clusterSet[str_inCluster])
                    left_rep_distance  = self.dissimilarityDistance(left_rep,clusterSet[str_inCluster])

                    if rep_distance == 0:
                        distances_vector[clusterSet[str_inCluster]] = 0
                    else:
                        distances_vector[clusterSet[str_inCluster]] = (right_rep_distance**2-rep_distance**2-left_rep_distance**2 ) / (2*rep_distance)

        else:
            if left_rep != None and right_rep == None:
                distances_vector[left_rep] = left_rep
            elif right_rep != None and left_rep == None:
                distances_vector[right_rep] = right_rep
            elif left_rep != None and right_rep != None:
                distances_vector[right_rep] = right_rep
            elif left_rep == None and right_rep == None:
                return None
                
        return distances_vector

    def median(self, distances):
        '''
        Returns the median value of a vector
        '''
        keys = list(distances.keys())
        if keys == 1:
            return keys[0]

        keys = list(distances.keys())
        median_position = int(len(keys)/2)
        median_value = keys[median_position]

        return median_value

    def OptimizeClusterSelection(self,Prototypes,numOfPrototypes):

        notwantedPrototypes = []
        for pr_1 in range(0,numOfPrototypes):
            for pr_2 in range(pr_1+1,numOfPrototypes):
                if self.dissimilarityDistance(Prototypes[pr_1],Prototypes[pr_2]) < self.prototypesFilterThr:
                    notwantedPrototypes.append(Prototypes[pr_2])

        newPrototypes = list((set(Prototypes)).difference(set(notwantedPrototypes)))
        
        if self.verboseLevel > 1:
            print("Prototypes before:")
            print(Prototypes)
            print("Not wanted:")
            print(set(notwantedPrototypes) )
            print("Final:")
            print(newPrototypes)

        return newPrototypes,len(newPrototypes)


    #####################################################################
    #       2. Embeddings based on the Vantage objects                  #
    #####################################################################

    '''
    CreateVantageEmbeddings(S,VantageObjects): Main function for creating the string embeddings based on the Vantage Objects
    '''
    def CreateVantageEmbeddings(self, S, VantageObjects, pairDictionary):

        # ------- Distance computing ------- #
        vectors = []
        for s in tqdm(range(0,S.size), desc="Creating embeddings", disable = self.disableTqdm, dynamic_ncols = True):
            string_embedding = []
            for p in range(0,VantageObjects.size):
                if VantageObjects[p] != None:
                    string_embedding.append(self.DistanceMetric(s,p,S,VantageObjects, pairDictionary))

            # --- Ranking representation ---- #
            ranked_string_embedding = stats.rankdata(string_embedding, method='min')

            # ------- Vectors dataset ------- #
            vectors.append(ranked_string_embedding)

        return np.array(vectors)


    '''
    DistanceMetric(s,p,S,Prototypes): Embedding method used for creating the space of objects
    '''
    def DistanceMetric(self, s, p, S, VantageObjects, pairDictionary):

        if self.distanceMetricEmbedding == 'l_inf':
            return self.l_inf(VantageObjects,S,s,p)
        elif self.distanceMetricEmbedding == 'edit':
            return self.dissimilarityDistance(S[s],VantageObjects[p])
        elif self.distanceMetricEmbedding == 'jaccard':
            return jaccard_distance(self.S_set[S[s]],self.S_set[VantageObjects[p]])
        elif self.distanceMetricEmbedding == 'euclid_jaccard':
            return self.hybrid_euclidJaccard(self.S_set[S[s]],self.S_set[VantageObjects[p]])
        else:
            warnings.warn("Available metrics: edit,jaccard,l_inf")


    def dropNone(array):
        array = list(filter(None, list(array)))
        return np.array(array)
    
    def l_inf(self,VantageObjects,S,s,p):
        max_distance = None
        for pp in range(0,VantageObjects.size):
            if VantageObjects[pp] != None:
                string_distance = self.dissimilarityDistance(S[s],VantageObjects[pp])    # distance String-i -> Vantage Object
                VO_distance     = self.dissimilarityDistance(VantageObjects[p],VantageObjects[pp])    # distance Vantage Object-j -> Vantage Object-i

                abs_diff = abs(string_distance-VO_distance)

                # --- Max distance diff --- #
                if max_distance == None:
                    max_distance = abs_diff
                elif abs_diff > max_distance:
                    max_distance = abs_diff
                    
        return max_distance
    
    def hybrid_euclidJaccard(self,s,p): 
        return math.sqrt(jaccard_distance(s,p))
    
    #####################################################################
    #                 3. Similarity checking                            #
    #####################################################################

    def SimilarityEvaluation(self, buckets,vectors,threshold,maxOnly=None,metric=None):

        numOfVectors = vectors.shape[0]
        vectorDim    = vectors.shape[1]
        mapping_matrix = np.zeros([numOfVectors,numOfVectors],dtype=np.int8)
        self.similarityProb_matrix = np.empty([numOfVectors,numOfVectors],dtype=np.float)* np.nan
        mapping = {}
        
        self.numOfComparisons = 0
        self.diffObjectsComparedSuccess = 0
        self.sameObjectsCompared = 0
        self.sameObjectsComparedSuccess = 0
        
        # Loop for every bucket
        for bucketid in tqdm(buckets.keys(), desc="Similarity checking", disable = self.disableTqdm, dynamic_ncols = True):
            bucket_vectors = buckets[bucketid]
            numOfVectors = len(bucket_vectors)
            
            if self.verboseLevel > 0:
                print(bucket_vectors)
            
            # For every vector inside the bucket
            for v_index in range(0,numOfVectors,1):
                v_vector_id = bucket_vectors[v_index]
                # Loop to all the other
                for i_index in range(v_index+1,numOfVectors,1):
                    i_vector_id = bucket_vectors[i_index]
                    if vectorDim == 1:
                        warnings.warn("Vector dim equal to 1- Setting metric to kendalltau")
                        metric = 'kendal'
                    
                    self.numOfComparisons+=1

                    if self.numOfComparisons >= 250000:
                        return None, None
                    
                    if metric == None or metric == 'kendal':  # Simple Kendal tau metric
                        similarity_prob, p_value = kendalltau(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'customKendal':  # Custom Kendal tau
                        numOf_discordant_pairs = _kendall_dis(vectors[v_vector_id].astype('intp'), vectors[i_vector_id].astype('intp'))
                        similarity_prob = (2*numOf_discordant_pairs) / (vectorDim*(vectorDim-1))
                    elif metric == 'jaccard':
                        similarity_prob = jaccard_score(vectors[v_vector_id], vectors[i_vector_id], average='micro')
                    elif metric == 'cosine':
                        similarity_prob = cosine_similarity(np.array(vectors[v_vector_id]).reshape(1, -1), np.array(vectors[i_vector_id]).reshape(1, -1))
                    elif metric == 'pearson':
                        similarity_prob, _ = pearsonr(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'spearman':
                        similarity_prob, _ = spearmanr(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'spearmanf':
                        similarity_prob = 1-spearman_footrule_distance(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'hamming':
                        similarity_prob, _ = hamming(vectors[v_vector_id].astype('intp'), vectors[i_vector_id].astype('intp'))
                    elif metric == 'kruskal':
                        if np.array_equal(vectors[v_vector_id],vectors[i_vector_id]):
                            similarity_prob=1.0
                        else:
                            _,similarity_prob = kruskal(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'ndcg_score':
                        similarity_prob, _ = ndcg_score(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'rbo':
                        similarity_prob = rbo(vectors[v_vector_id], vectors[i_vector_id], self.rbo_p)
                    elif metric == 'wta':
                        similarity_prob = wtaSimilarity(vectors[v_vector_id], vectors[i_vector_id])
                    elif metric == 'mannwhitneyu':
                        if np.array_equal(vectors[v_vector_id],vectors[i_vector_id]):
                            similarity_prob=1.0
                        else:
                            _,similarity_prob = mannwhitneyu(vectors[v_vector_id], vectors[i_vector_id])
                    else:
                        warnings.warn("SimilarityEvaluation: Available similarity metrics: kendal,customKendal,jaccard,ndcg_score,cosine,spearman,pearson")


                    self.similarityProb_matrix[v_vector_id][i_vector_id] = similarity_prob
                    self.similarityProb_matrix[i_vector_id][v_vector_id] = similarity_prob
                    
                    if self.true_matrix[v_vector_id][i_vector_id] or self.true_matrix[i_vector_id][v_vector_id]:
                        self.sameObjectsCompared += 1

                    if similarity_prob > threshold:
                        if v_vector_id not in mapping.keys():
                            mapping[v_vector_id] = []
                        mapping[v_vector_id].append(i_vector_id)  # insert into mapping
                        mapping_matrix[v_vector_id][i_vector_id] = 1  # inform prediction matrix
                        mapping_matrix[i_vector_id][v_vector_id] = 1  # inform prediction matrix
                        if self.true_matrix[v_vector_id][i_vector_id] or self.true_matrix[i_vector_id][v_vector_id]:
                            self.sameObjectsComparedSuccess += 1
                    elif similarity_prob <= threshold and self.true_matrix[v_vector_id][i_vector_id] == 0 and self.true_matrix[i_vector_id][v_vector_id] == 0:
                        self.diffObjectsComparedSuccess += 1


        return mapping, np.triu(mapping_matrix)

    
    
    #####################################################################
    #                          Evaluation                               # 
    #####################################################################

    def evaluate(self, predicted_matrix, true_matrix, with_classification_report=False, with_confusion_matrix=False, with_detailed_report=False):
        
        if self.verboseLevel >= 0:
            print("#####################################################################\n#                          Evaluation                               #\n#####################################################################\n")
        transformToVector = np.triu_indices(len(true_matrix))    
        true_matrix = true_matrix[transformToVector]
        predicted_matrix = predicted_matrix[transformToVector]
        
        acc = 100*accuracy_score(true_matrix, predicted_matrix)
        f1 =  100*f1_score(true_matrix, predicted_matrix)
        recall = 100*recall_score(true_matrix, predicted_matrix)
        precision = 100*precision_score(true_matrix, predicted_matrix)

        if self.verboseLevel >= 0:
            print("Accuracy:  %3.2f %%" % (acc))
            print("F1-Score:  %3.2f %%" % (f1))
            print("Recall:    %3.2f %%" % (recall))
            print("Precision: %3.2f %%" % (precision))

        if with_classification_report:
            print("\nClassification report:\n")
            print(classification_report(true_matrix, predicted_matrix))
            print('\n')
        
        if with_confusion_matrix:
            cm = sklearn.metrics.confusion_matrix(true_matrix,predicted_matrix,labels=[0,1])
            create_ConfusionMatrix(cm,'Confusion matrix')
            print('\n\n')

        if with_detailed_report:
            report(self)
        
        return acc,f1,precision,recall

def report(model):
    
    print("-----------------------\n--- DETAILED REPORT ---\n-----------------------\n")
    print("\n> 1. Prototype selection\n")
    print("\n> 2. Embedding phase\n")
    print("\n> 3. WTA hashing\n")
    print("Number of buckets created ", len(model.buckets.keys()))
    for key in model.buckets.keys():
        print(key," -> ", len(model.buckets[key]))
        
    print("\n> 4. Similarity checking\n")
    print("Total comparisons: ", model.numOfComparisons)
    print(" -> between same objects: ", model.sameObjectsCompared )
    print(" -> between same objects with success: ", model.sameObjectsComparedSuccess)
    print(" -> between different objects with success: ", model.diffObjectsComparedSuccess )
    

def customClassificationReport(predicted_matrix, true_matrix):
    
    size = len(predicted_matrix)
    true_positives = 0; false_positives = 0; true_negatives = 0; false_negatives = 0;
    i=0
    while i < size:
        j = i
        while j < size:
            if predicted_matrix[i][j] == true_matrix[i][j] == 1:
                true_positives += 1
            elif predicted_matrix[i][j] == true_matrix[i][j] == 0:
                true_negatives += 1
            elif predicted_matrix[i][j] == 0 and true_matrix[i][j] == 1:
                false_negatives += 1
            elif predicted_matrix[i][j] == 1 and true_matrix[i][j] == 0:
                false_positives += 1            
            j += 1
        i += 1
        
    accuracy = ((true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives))*100
    precision = (true_positives / (true_positives + false_positives))*100
    recall = (true_positives / (true_positives + false_negatives))*100
    f1 = 2*((precision*recall)/(precision+recall))
    
    print("Accuracy:  %.2f %%" % (accuracy))
    print("F1-Score:  %.2f %%" % (f1))
    print("Recall:    %.2f %%" % (recall))
    print("Precision: %.2f %%" % (precision))
    
    print("True positives:  ", true_positives)
    print("True negatives:  ", true_negatives)
    print("False positives: ", false_positives)
    print("False negatives: ", false_negatives)

def set_params(params_dict):

    max_numberOf_clusters = params_dict["max_numberOf_clusters"]
    max_dissimilarityDistance = params_dict["max_dissimilarityDistance"]
    windowSize = params_dict["windowSize"]
    similarityThreshold = params_dict["similarityThreshold"]
    metric = params_dict["metric"]
    similarityVectors = params_dict["similarityVectors"]
    number_of_permutations = params_dict["number_of_permutations"]
    distanceMetric = params_dict["distanceMetric"]
    distanceMetricEmbedding = params_dict["distanceMetricEmbedding"]
    ngramms = params_dict["ngramms"]
    jaccard_withchars =  params_dict["jaccard_withchars"]
    prototypesFilterThr = params_dict["prototypesFilterThr"]
    # rbo_p = params_dict["rbo_p"]
    wtaM = params_dict["wtaM"]

    modelCreated = RankedWTAHash(
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
        # rbo_p = rbo_p,
        wtaM = wtaM
    )

    return modelCreated