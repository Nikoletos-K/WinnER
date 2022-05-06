'''
@author: Konstantinos Nikoletos, 2022
'''
from asyncio.log import logger
import numpy as np
import editdistance
import sklearn
import time
import warnings
import nltk
import math
import bloom_filter
import multiprocessing
import numba
import threading
import logging
import os, sys

from joblib import Parallel, delayed

# from tqdm.notebook import tqdm as tqdm
from tqdm import tqdm as tqdm

from scipy.spatial.distance import hamming
from scipy.stats._stats import _kendall_dis
from scipy.stats import spearmanr,kendalltau,pearsonr,kruskal,mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import jaccard_distance
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, recall_score, precision_score, classification_report
from scipy import stats 
from scipy.spatial.distance import hamming, jaccard
from sklearn.metrics import ndcg_score
from bloom_filter import BloomFilter
from cantor import q_encode

# --------------------------------- #
# ---- Import from local files ---- #
# --------------------------------- #

from hash.wta import WTA, wta_similarity
from plot.dimension_reduction import SpaceVisualization2D, SpaceVisualization3D, SpaceVisualizationEmbeddings3D
from plot.heatmap import myHeatmap
from plot.confusion_matrix import create_ConfusionMatrix
# from utils.metrics import *
from utils.tokenizer import Tokenizer
from utils.error_messages import *

# --------------------------------- #
# ----    Main model class     ---- #
# --------------------------------- #

class WinnER:

    _data_size = 0
    _string_array = None
    _pair_dict = dict()
    _string_index_map = None
    _input_strings = None
    _num_of_comparisons = 0
    _labels_ground_truth = None 
    _true_matrix = None
    _gt_provided = False

    def __init__(self, 
            max_num_of_clusters = None, 
            max_dissimilarity_distance = None, 
            window_size = None,   
            number_of_permutations = 1, 
            char_tokenization = True,
            embedding_distance_metric = 'euclid_jaccard', 
            metric = 'kendal', 
            similarity_vectors = 'ranked', 
            distance_metric = 'euclid_jaccard', 
            prototypes_optimization_thr = None, 
            ngrams = 3,
            similarity_threshold = None, 
            num_of_threads = 16,
            verbose_level = 0, 
            rbo_p = 0.7, 
            wta_m = 1, 
            max_num_of_comparisons = 250000, 
            disable_tqdm = False,
            enable_blocking = True,
            debug_stop = None   
        ):
        '''
          Constructor
        '''

        # Model hyper-parameters
        self.max_num_of_clusters = max_num_of_clusters
        self.max_dissimilarity_distance = max_dissimilarity_distance
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold


        self.metric = metric
        self.similarity_vectors = similarity_vectors
        self.number_of_permutations = number_of_permutations
        self.distance_metric = distance_metric
        self.embedding_distance_metric = embedding_distance_metric
        self.ngrams = ngrams
        self.char_tokenization =  char_tokenization
        if prototypes_optimization_thr == None:
            self.prototypes_optimization_thr = max_dissimilarity_distance
        else:
            self.prototypes_optimization_thr = prototypes_optimization_thr
        self.selection_variance = None
        self._num_of_comparisons = 0
        self.verbose_level = verbose_level
        self.rbo_p = rbo_p
        self.wta_m = wta_m
        self.MAX_NUMBER_OF_COMPARISONS = max_num_of_comparisons
        self.disable_tqdm = disable_tqdm
        self.num_of_threads = num_of_threads
        self.enable_blocking = enable_blocking
        self.debug_stop = debug_stop
        
    def groundtruth(self, labels_ground_truth, true_matrix):

        self._labels_ground_truth = labels_ground_truth
        self._true_matrix = true_matrix
        self._gt_provided = True

    def fit(self, input_strings):
        """
          Models main method
          ---
          Parameters:
          - input_strings : ER data
          
          ### Returns
          self : The fitted model.
        """

        if self.verbose_level >=0 :
            print("\n#####################################################################\n#                           .~  WinnER  ~.                          #\n#####################################################################\n")

        _data_size = len(input_strings)

        self._input_strings = np.array(input_strings,dtype=object)
        tok = Tokenizer(self.ngrams, self.char_tokenization)
        self._string_array = tok.process(input_strings)
        self._string_index_map = np.arange(0,len(input_strings),1)
        self.bloom_filter_size = _data_size*_data_size

        if self.verbose_level > 1:
            print("\n\nString positions are:")
            print(self._string_index_map)
            print("\n")

        if self.verbose_level >=0 :
            print("###########################################################\n# > 1. Prototype selection phase                          #\n###########################################################\n")
            print("\n-> Finding prototypes and representatives of each cluster:")
        
        prototypes_time = time.time()
        self.prototype_array, self.num_of_prototypes = self.PrototypeSelection(self._string_index_map,self.max_num_of_clusters, self.max_dissimilarity_distance)
        self.embedding_dimension = self.prototype_array.size
        
        if self.verbose_level >= 1:

            if self.verbose_level >= 2:
                print("\n- Prototypes selected:")
                print(self.prototype_array)
            heatmap_data = []
            for pr in self.prototype_array:
                if self.verbose_level >= 2:
                    print(pr," -> ",self._input_strings[pr])
                heatmap_data.append(self._string_array[pr])            
            if self.num_of_prototypes > 2:
                self.selection_variance = myHeatmap(self.prototype_array,self.metric,self.dissimilarity_distance)
                print("\n- Mean variance in prototype selection: ", self.selection_variance)

        self.prototypes_time = time.time() - prototypes_time
        
        if self.verbose_level >=0 :
            print("\n- Final number of prototypes: ",self.num_of_prototypes )
            print("\n# Finished in %.6s secs" % (prototypes_time))
            print("\n")
        
        if self.verbose_level >= 0:
            print("###########################################################\n# > 2. Embeddings based on the Vantage objects            #\n###########################################################\n")
            print("\n-> Creating Embeddings:")
        embeddings_time = time.time()
        self.embeddings = self.CreateVantageEmbeddings(self._string_index_map, self.prototype_array)
       
        if self.verbose_level > 0:
            SpaceVisualization2D(self.embeddings, self.prototype_array)        
        
        self.embeddings_time = time.time() - embeddings_time

        if self.verbose_level >=0:
            print("\n# Finished in %.6s secs" % (embeddings_time))
            print("\n")

        if self.enable_blocking:

            if self.verbose_level >= 0:
                print("###########################################################\n# > 3. WTA Hashing                                        #\n###########################################################\n")
                print("\n-> Creating WTA Buckets:")

            wta_time = time.time()
            wta = WTA(self.window_size, self.number_of_permutations, self.wta_m, self.disable_tqdm)
            self.buckets, self.ranked_vectors = wta.hash(self.embeddings)
            
            if self.verbose_level > 1:
                print("- WTA buckets: ")
                for key in self.buckets.keys():
                    print(key," -> ",self.buckets[key])
            
            if self.verbose_level >= 0:
                print("\n- WTA number of buckets: ", len(self.buckets.keys()))
            
            if self.verbose_level > 1:
                print("\n- WTA RankedVectors after permutation:")
                print(self.ranked_vectors)

            if self.verbose_level > 0:
                if self._gt_provided and self.similarity_vectors == 'ranked':
                    SpaceVisualizationEmbeddings3D(self.ranked_vectors, self._labels_ground_truth)
                elif self._gt_provided and self.similarity_vectors == 'initial':
                    SpaceVisualizationEmbeddings3D(self.embeddings, self._labels_ground_truth)

            self.wta_time = time.time() - wta_time

            if self.debug_stop:
                sys.exit()

            if self.verbose_level >=0 :
                print("\n# Finished in %.6s secs" % (wta_time))
                print("\n")
        
        if self.verbose_level >=0 :
            print("###########################################################\n# > 4. Similarity checking                                #\n###########################################################\n")
            print("\n-> Similarity checking:")

        similarity_time = time.time()

        if self.enable_blocking:
            if self.similarity_vectors == 'ranked':
                self.mapping, self.mapping_matrix = self.SimilarityEvaluation(self.buckets, self.ranked_vectors)
            elif self.similarity_vectors == 'initial':
                self.mapping, self.mapping_matrix = self.SimilarityEvaluation(self.buckets, self.embeddings)
            else:
                warnings.warn("similarity_vectors: Available options are: ranked, initial")

            if self.mapping == None and self.mapping_matrix == None:
                return None
        else:
            print(self.embeddings)
            self.mapping, self.mapping_matrix = self.SimilarityEvaluationWithoutHashing(self.embeddings)

        if self.verbose_level > 1:
            print("- Similarity mapping in a matrix")
            print(self.mapping_matrix)
        
        if self.verbose_level > 0:
            print("Total comparisons: ", self._num_of_comparisons)
            print(" -> between same objects: ", self.sameObjectsCompared )
            print(" -> between same objects with success: ", self.sameObjectsComparedSuccess)
            print(" -> between different objects: ", self.difObjectsCompared)
            print(" -> between different objects with success: ", self.diffObjectsComparedSuccess)
        
        self.similarity_time = time.time() - similarity_time

        if self.verbose_level >=0 :
            print("\n# Finished in %.6s secs" % (similarity_time))
            print("\n#####################################################################\n#                           .~  End  ~.                             #\n#####################################################################\n")

        return self

    def dissimilarity_distance(self, str1, str2):

        if self.verbose_level > 2:
            print("-> ", self._input_strings[str1])
            print("--> ", self._input_strings[str2])

        if frozenset([str1,str2]) in self._pair_dict.keys():
            return self._pair_dict[frozenset([str1,str2])]
        else:
            if self.distance_metric == 'edit':
                distance = editdistance.eval(self._string_array[str1],self._string_array[str2])
            elif self.distance_metric == 'jaccard':
                distance = jaccard_distance(self._string_array[str1],self._string_array[str2])
            elif self.distance_metric == 'euclid_jaccard':
                distance = math.sqrt(jaccard_distance(self._string_array[str1],self._string_array[str2]))                
            else:
                warnings.warn("Available metrics for space creation: edit, jaccard, euclid_jaccard ")

            self._pair_dict[frozenset([str1,str2])] = distance
            
            if self.verbose_level > 2:
                print(distance)
            
            return distance

    #####################################################################
    # 1. Prototype selection algorithm                                  #
    #####################################################################

    '''
    PrototypeSelection(S,k,d) 
    The String Clustering and Prototype Selection Algorithm
    is the main clustering method, that takes as input the intial strings S, 
    the max number of clusters to be generated in k,
    the maximum allowable distance of a string to join a cluster in var d
    and returns the prototype for each cluster in array Prototype
    '''
    def PrototypeSelection(self,S,k,d):

        # ----------------- Initialization phase ----------------- #
        i = 0
        j = 0
        C = np.empty([S.size], dtype=int)
        r = np.empty([2,k],dtype=object)

        Clusters = [ [] for l in range(0,k)]

        for i in tqdm(range(0,S.size,1), desc="Representatives selection", disable = self.disable_tqdm, dynamic_ncols = True):     # String-clustering phase, for all strings
            while j < k :       # iteration through clusters, for all clusters
                if r[0][j] == None:      # case empty first representative for cluster j
                    r[0][j] = S[i]   # init cluster representative with string i
                    C[i] = j         # store in C that i-string belongs to cluster j
                    Clusters[j].append(S[i])
                    break
                elif r[1][j] == None and (self.dissimilarity_distance(S[i],r[0][j]) <= d):  # case empty second representative
                    r[1][j] = S[i]                                             # and ED of representative 1  smaller than i-th string
                    C[i] = j
                    Clusters[j].append(S[i])
                    break
                elif (r[0][j] != None and r[1][j] != None) and (self.dissimilarity_distance(S[i],r[0][j]) + self.dissimilarity_distance(S[i],r[1][j])) <= d:
                    C[i] = j
                    Clusters[j].append(S[i])
                    break
                else:
                    j += 1
            i += 1

        # ----------------- Prototype selection phase ----------------- #

        Projections = []
        Prototypes = []
        sortedProjections = []

        if self.verbose_level > 2:
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
        for j in tqdm(range(0,k,1), desc="Prototype selection", disable = self.disable_tqdm, dynamic_ncols = True):
            
            apprxDistances = self.ApproximatedProjectionDistancesofCluster(r[1][j], r[0][j], Clusters[j])
            
            if apprxDistances == None:
                new_numofClusters-=1
                continue
            
            Projections.append(apprxDistances)
            sortedProjections.append({new_numofClusters: v for new_numofClusters, v in sorted(Projections[prototype_index].items(), key=lambda item: item[1])})
            Prototypes.append(self.median(sortedProjections[prototype_index]))
            prototype_index += 1
        
        Prototypes, new_numofClusters = self.OptimizeClusterSelection(Prototypes, new_numofClusters)

        
        return np.array(Prototypes), new_numofClusters


    def ApproximatedProjectionDistancesofCluster(self, right_rep, left_rep, clusterSet):

        distances_vector = dict()

        if len(clusterSet) > 2:
            rep_distance = self.dissimilarity_distance(right_rep,left_rep)

            for str_inCluster in range(0, len(clusterSet)):
                if clusterSet[str_inCluster] != right_rep and clusterSet[str_inCluster] != left_rep:
                    right_rep_distance = self.dissimilarity_distance(right_rep,clusterSet[str_inCluster])
                    left_rep_distance  = self.dissimilarity_distance(left_rep,clusterSet[str_inCluster])

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
        for pr_1 in tqdm(range(0,numOfPrototypes), desc="Prototype optimization", disable = self.disable_tqdm, dynamic_ncols = True):
            for pr_2 in range(pr_1+1,numOfPrototypes):
                if self.dissimilarity_distance(Prototypes[pr_1],Prototypes[pr_2]) < self.prototypes_optimization_thr:
                    notwantedPrototypes.append(Prototypes[pr_2])

        newPrototypes = list((set(Prototypes)).difference(set(notwantedPrototypes)))
        
        if self.verbose_level > 1:
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
    def CreateVantageEmbeddings(self, S, VantageObjects):

        # ------- Distance computing ------- #
        vectors = []
        for s in tqdm(range(0,S.size), desc="Creating embeddings", disable = self.disable_tqdm, dynamic_ncols = True):
            string_embedding = []
            for p in range(0,VantageObjects.size):
                if VantageObjects[p] != None:
                    string_embedding.append(self.DistanceMetric(s, p, S, VantageObjects))

            # --- Ranking representation ---- #
            ranked_string_embedding = stats.rankdata(string_embedding, method='min')

            # ------- Vectors dataset ------- #
            vectors.append(ranked_string_embedding)

        return np.array(vectors)


    '''
    DistanceMetric(s,p,S,Prototypes): Embedding method used for creating the space of objects
    '''
    def DistanceMetric(self, s, p, S, VantageObjects):

        if self.embedding_distance_metric == 'l_inf':
            return self.l_inf(VantageObjects,S,s,p)
        elif self.embedding_distance_metric == 'edit':
            return self.dissimilarity_distance(S[s],VantageObjects[p])
        elif self.embedding_distance_metric == 'jaccard':
            return jaccard_distance(self._string_array[S[s]],self._string_array[VantageObjects[p]])
        elif self.embedding_distance_metric == 'euclid_jaccard':
            return self.hybridEuclidJaccard(self._string_array[S[s]],self._string_array[VantageObjects[p]])
        else:
            warnings.warn("Available metrics: edit, jaccard, euclid_jaccard, l_inf")
    
    def l_inf(self,VantageObjects,S,s,p):
        max_distance = None
        for pp in range(0,VantageObjects.size):
            if VantageObjects[pp] != None:
                string_distance = self.dissimilarity_distance(S[s],VantageObjects[pp])    # distance String-i -> Vantage Object
                VO_distance     = self.dissimilarity_distance(VantageObjects[p],VantageObjects[pp])    # distance Vantage Object-j -> Vantage Object-i

                abs_diff = abs(string_distance-VO_distance)

                # --- Max distance diff --- #
                if max_distance == None:
                    max_distance = abs_diff
                elif abs_diff > max_distance:
                    max_distance = abs_diff
                    
        return max_distance
    
    def hybridEuclidJaccard(self,s,p): 
        return math.sqrt(jaccard_distance(s,p))
    
    #####################################################################
    #                 3. Similarity checking                            #
    #####################################################################

    def SimilarityEvaluationWithoutHashing(self, vectors):

        num_of_vectors = vectors.shape[0]
        vector_x_dimension = vectors.shape[1]
        self.mapping_matrix = np.zeros([num_of_vectors,num_of_vectors],dtype=np.int8)
        self.similarityProb_matrix = np.empty([num_of_vectors,num_of_vectors],dtype=np.float)* np.nan
        self.mapping = {}
        
        self._num_of_comparisons = 0
        self.diffObjectsComparedSuccess = 0
        self.difObjectsCompared = 0
        self.sameObjectsCompared = 0
        self.sameObjectsComparedSuccess = 0
        self.bloomFilter = BloomFilter(max_elements = self.bloom_filter_size, error_rate=0.1)

        metric = self.metric

        for x in range(0, num_of_vectors, 1):
            
            # Loop to all the other
            for y in range(x + 1, num_of_vectors, 1):
                v_vector_id = x
                i_vector_id = y

                cantor_unique_index = q_encode(x, y)
                if cantor_unique_index in self.bloomFilter:
                    continue
                else:
                    self.bloomFilter.add(cantor_unique_index)

                self._num_of_comparisons += 1

                if self._num_of_comparisons >= self.MAX_NUMBER_OF_COMPARISONS:
                    warnings.warn("Upper bound of comparisons has been achieved", DeprecationWarning)
                
                if metric == None or metric == 'kendal':  # Simple Kendal tau metric
                    similarity_prob, p_value = kendalltau(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'customKendal':  # Custom Kendal tau
                    numOf_discordant_pairs = _kendall_dis(vectors[v_vector_id].astype('intp'), vectors[i_vector_id].astype('intp'))
                    similarity_prob = (2*numOf_discordant_pairs) / (vector_x_dimension*(vector_x_dimension-1))
                elif metric == 'jaccard':
                    similarity_prob = jaccard_score(vectors[v_vector_id], vectors[i_vector_id], average='micro')
                elif metric == 'cosine':
                    similarity_prob = cosine_similarity(np.array(vectors[v_vector_id]).reshape(1, -1), np.array(vectors[i_vector_id]).reshape(1, -1))
                elif metric == 'pearson':
                    similarity_prob, _ = pearsonr(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'spearman':
                    similarity_prob, _ = spearmanr(vectors[v_vector_id], vectors[i_vector_id], nan_policy='omit')
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
                    similarity_prob = wta_similarity(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'mannwhitneyu':
                    if np.array_equal(vectors[v_vector_id],vectors[i_vector_id]):
                        similarity_prob=1.0
                    else:
                        _,similarity_prob = mannwhitneyu(vectors[v_vector_id], vectors[i_vector_id])
                else:
                    warnings.warn("Similarity not exists, available similarity metrics: kendal, rbo, spearman, pearson")

                self.similarityProb_matrix[v_vector_id][i_vector_id] = similarity_prob
                self.similarityProb_matrix[i_vector_id][v_vector_id] = similarity_prob
                
                if self._true_matrix is not None: 
                    
                    if self._true_matrix[v_vector_id][i_vector_id] or self._true_matrix[i_vector_id][v_vector_id]:
                        self.sameObjectsCompared += 1
                    
                    if self._true_matrix[v_vector_id][i_vector_id] == 0 or self._true_matrix[i_vector_id][v_vector_id] == 0:
                        self.difObjectsCompared += 1
                
                if similarity_prob > self.similarity_threshold:
                    if v_vector_id not in self.mapping.keys():
                        self.mapping[v_vector_id] = []
                    self.mapping[v_vector_id].append(i_vector_id)  # insert into mapping
                    self.mapping_matrix[v_vector_id][i_vector_id] = 1  # inform prediction matrix
                    self.mapping_matrix[i_vector_id][v_vector_id] = 1  # inform prediction matrix

                    if (self._true_matrix is not None) and  self._true_matrix[v_vector_id][i_vector_id] or self._true_matrix[i_vector_id][v_vector_id]:
                        self.sameObjectsComparedSuccess += 1

                elif similarity_prob <= self.similarity_threshold and self._true_matrix is not None and self._true_matrix[v_vector_id][i_vector_id] == 0 and self._true_matrix[i_vector_id][v_vector_id] == 0:
                    self.diffObjectsComparedSuccess += 1

        return self.mapping, np.triu(self.mapping_matrix)


    def SimilarityEvaluationBucket(self, bucket_vectors, lock):
        logging.info('Bucket checking')
        metric = self.metric
        vectors = self.embeddings
        vector_x_dimension = vectors.shape[1]
        num_of_vectors = len(bucket_vectors)
        for v_index in range(0,num_of_vectors,1):
            v_vector_id = bucket_vectors[v_index]

            # Loop to all the other
            for i_index in range(v_index+1,num_of_vectors,1):
                i_vector_id = bucket_vectors[i_index]

                cantor_unique_index = q_encode(v_vector_id, i_vector_id)
                if cantor_unique_index in self.bloomFilter:
                    continue
                else:
                    lock.acquire()
                    self.bloomFilter.add(cantor_unique_index)
                    lock.release()

                lock.acquire()
                self._num_of_comparisons += 1
                lock.release()

                if self._num_of_comparisons >= self.MAX_NUMBER_OF_COMPARISONS:
                    warnings.warn("Upper bound of comparisons has been achieved", DeprecationWarning)
                
                if metric == None or metric == 'kendal':  # Simple Kendal tau metric
                    similarity_prob, p_value = kendalltau(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'customKendal':  # Custom Kendal tau
                    numOf_discordant_pairs = _kendall_dis(vectors[v_vector_id].astype('intp'), vectors[i_vector_id].astype('intp'))
                    similarity_prob = (2*numOf_discordant_pairs) / (vector_x_dimension*(vector_x_dimension-1))
                elif metric == 'jaccard':
                    similarity_prob = jaccard_score(vectors[v_vector_id], vectors[i_vector_id], average='micro')
                elif metric == 'cosine':
                    similarity_prob = cosine_similarity(np.array(vectors[v_vector_id]).reshape(1, -1), np.array(vectors[i_vector_id]).reshape(1, -1))
                elif metric == 'pearson':
                    similarity_prob, _ = pearsonr(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'spearman':
                    similarity_prob, _ = spearmanr(vectors[v_vector_id], vectors[i_vector_id], nan_policy='omit')
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
                    similarity_prob = wta_similarity(vectors[v_vector_id], vectors[i_vector_id])
                elif metric == 'mannwhitneyu':
                    if np.array_equal(vectors[v_vector_id],vectors[i_vector_id]):
                        similarity_prob=1.0
                    else:
                        _,similarity_prob = mannwhitneyu(vectors[v_vector_id], vectors[i_vector_id])
                else:
                    warnings.warn("Similarity not exists, available similarity metrics: kendal, rbo, spearman, pearson")


                lock.acquire()
                self.similarityProb_matrix[v_vector_id][i_vector_id] = similarity_prob
                self.similarityProb_matrix[i_vector_id][v_vector_id] = similarity_prob
                
                if self._true_matrix is not None: 
                    
                    if self._true_matrix[v_vector_id][i_vector_id] or self._true_matrix[i_vector_id][v_vector_id]:
                        self.sameObjectsCompared += 1
                    
                    if self._true_matrix[v_vector_id][i_vector_id] == 0 or self._true_matrix[i_vector_id][v_vector_id] == 0:
                        self.difObjectsCompared += 1

                if similarity_prob > self.similarity_threshold:
                    if v_vector_id not in self.mapping.keys():
                        self.mapping[v_vector_id] = []
                    self.mapping[v_vector_id].append(i_vector_id)  # insert into mapping
                    self.mapping_matrix[v_vector_id][i_vector_id] = 1  # inform prediction matrix
                    self.mapping_matrix[i_vector_id][v_vector_id] = 1  # inform prediction matrix
                    if self._true_matrix is not None and self._true_matrix[v_vector_id][i_vector_id] or self._true_matrix[i_vector_id][v_vector_id]:
                        self.sameObjectsComparedSuccess += 1
                elif similarity_prob <= self.similarity_threshold and self._true_matrix[v_vector_id][i_vector_id] == 0 and self._true_matrix[i_vector_id][v_vector_id] == 0:
                    self.diffObjectsComparedSuccess += 1
                lock.release()


    def SimilarityEvaluation(self, buckets, vectors):

        num_of_vectors = vectors.shape[0]
        self.mapping_matrix = np.zeros([num_of_vectors,num_of_vectors],dtype=np.int8)
        self.similarityProb_matrix = np.empty([num_of_vectors,num_of_vectors],dtype=np.float)* np.nan
        self.mapping = {}
        
        self._num_of_comparisons = 0
        self.diffObjectsComparedSuccess = 0
        self.difObjectsCompared = 0
        self.sameObjectsCompared = 0
        self.sameObjectsComparedSuccess = 0
        self.bloomFilter = BloomFilter(max_elements = self.bloom_filter_size, error_rate=0.1)
        self.numOfBuckets = len(buckets.keys())

        # Loop for every bucket
        lock = threading.Lock()
        # lock = multiprocessing.Lock()
        
        thread_index = 0
        thread_pool = []
        
        for bucketid, thread_index in tqdm(zip(buckets.keys(), range(0, self.numOfBuckets, 1)), desc="Similarity checking", disable = self.disable_tqdm, total = self.numOfBuckets, dynamic_ncols = True):
            bucket_vectors = buckets[bucketid]

            if isinstance(bucket_vectors, set):
                bucket_vectors = list(bucket_vectors)
            
            if self.verbose_level > 1:
                print(bucket_vectors)

            if thread_index % self.num_of_threads == 0:
                [t.start() for t in thread_pool]            
                [t.join() for t in thread_pool]
                thread_pool = []

            thread_pool.append(
                threading.Thread(target = self.SimilarityEvaluationBucket, args=(bucket_vectors, lock))
            )

            # thread_pool.append(multiprocessing.Process(target = self.SimilarityEvaluationBucket,
            #                 args=(bucket_vectors, lock)))
        
        if self.numOfBuckets % self.num_of_threads != 0: 
            [t.start() for t in thread_pool]
            [t.join() for t in thread_pool]

        return self.mapping, np.triu(self.mapping_matrix)


    
    #####################################################################
    #                          Evaluation                               # 
    #####################################################################

    def evaluate(self, predicted_matrix, with_classification_report=False, with_confusion_matrix=True, with_detailed_report=False):
        
        if self.verbose_level >= 0:
            print("#####################################################################\n#                          Evaluation                               #\n#####################################################################\n")
        transformToVector = np.triu_indices(len(self._true_matrix))    
        true_matrix = self._true_matrix[transformToVector]
        predicted_matrix = predicted_matrix[transformToVector]
        
        acc = 100*accuracy_score(true_matrix, predicted_matrix)
        f1 =  100*f1_score(true_matrix, predicted_matrix)
        recall = 100*recall_score(true_matrix, predicted_matrix)
        precision = 100*precision_score(true_matrix, predicted_matrix)

        if self.verbose_level >= 0:
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

    #####################################################################
    #                          Utilities                                # 
    #####################################################################

    def get_params(self):
        return {
            "max_num_of_clusters" : self.max_num_of_clusters, 
            "max_dissimilarity_distance" : self.max_dissimilarity_distance, 
            "window_size" : self.window_size,  
            "number_of_permutations" : self.number_of_permutations, 
            "char_tokenization" : self.char_tokenization,
            "embedding_distance_metric" : self.embedding_distance_metric, 
            "metric" : self.metric, 
            "similarity_vectors" : self.similarity_vectors, 
            "distance_metric" : self.distance_metric, 
            "prototypes_optimization_thr" : self.prototypes_optimization_thr, 
            "ngrams" : self.ngrams, 
            "similarity_threshold" : self.similarity_threshold, 
            "num_of_threads" : self.num_of_threads,
            "verbose_level" : self.verbose_level, 
            "disable_tqdm" : self.disable_tqdm            
        }

    def set_params(self, params_dict):
        if "max_num_of_clusters" in params_dict.keys():
            self.max_num_of_clusters = params_dict["max_num_of_clusters"]
        if "max_dissimilarity_distance" in params_dict.keys():
            self.max_dissimilarity_distance = params_dict["max_dissimilarity_distance"]
            self.prototypes_optimization_thr = params_dict["max_dissimilarity_distance"]
        if "window_size" in params_dict.keys():
            self.window_size = params_dict["window_size"]
        if "similarity_threshold" in params_dict.keys():
            self.similarity_threshold = params_dict["similarity_threshold"]
        if "metric" in params_dict.keys():
            self.metric = params_dict["metric"]
        if "similarity_vectors" in params_dict.keys():
            self.similarity_vectors = params_dict["similarity_vectors"]
        if "number_of_permutations" in params_dict.keys():
            self.number_of_permutations = params_dict["number_of_permutations"]
        if "distance_metric" in params_dict.keys():
            self.distance_metric = params_dict["distance_metric"]
        if "embedding_distance_metric" in params_dict.keys():
            self.embedding_distance_metric = params_dict["embedding_distance_metric"]
        if "ngrams" in params_dict.keys():
            self.ngrams = params_dict["ngrams"]
        if "char_tokenization" in params_dict.keys():
            self.char_tokenization = params_dict["char_tokenization"]
        if "prototypes_optimization_thr" in params_dict.keys():
            self.prototypes_optimization_thr = params_dict["prototypes_optimization_thr"]
        if "verbose_level" in params_dict.keys():
            self.verbose_level = params_dict["verbose_level"]
    

def report(model):
    
    print("-----------------------\n--- DETAILED REPORT ---\n-----------------------\n")
    print("\n> 1. Prototype selection\n")
    print("\n> 2. Embedding phase\n")
    print("\n> 3. WTA hashing\n")
    print("Number of buckets created ", len(model.buckets.keys()))
    for key in model.buckets.keys():
        print(key," -> ", len(model.buckets[key]))
    print("\n> 4. Similarity checking\n")
    print("Total comparisons: ", model._num_of_comparisons)
    print(" -> between same objects: ", model.sameObjectsCompared )
    print(" -> between same objects with success: ", model.sameObjectsComparedSuccess)
    print(" -> between different objects: ", model.difObjectsCompared)
    print(" -> between different objects with success: ", model.diffObjectsComparedSuccess)
    

def custom_classification_report(predicted_matrix, true_matrix):
    
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


