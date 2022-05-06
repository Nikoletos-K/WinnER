'''
@author: Konstantinos Nikoletos, 2021
'''
import numpy as np
import warnings
from numpy.lib.twodim_base import diag
from tqdm.notebook import tqdm as tqdm
import logging

class WTA:
    
    def __init__(self, K, number_of_permutations, m=1, disable_tqdm=False):
        
        self.K = K
        self.number_of_permutations = number_of_permutations
        self.m = m
        self.disable_tqdm = disable_tqdm
    
    def hash(self, vectors):
        '''
        Winner Take All hash - Yagnik
        ---

          vectors: initial vectors
          K: window size
          number_of_permutations: number of times each vector will be permuted  
        '''

        new_vectors = []
        buckets = dict()
        num_of_vectors = vectors.shape[0]
        vector_dimension = vectors.shape[1]

        if vector_dimension < self.K:
            self.K = vector_dimension
            # logging.error("Window size greater than vector dimension")
            # warnings.warn("Window size greater than vector dimension")

        C = np.zeros([self.number_of_permutations * num_of_vectors], dtype=int)
        i=0;
        for permutation_index in tqdm(range(0,self.number_of_permutations,1), desc="WTA hashing", dynamic_ncols = True, disable = self.disable_tqdm):

            # randomization is without replacement and has to be consistent 
            # across all samples and hence the notion of permutations
            theta = np.random.permutation(vector_dimension) 

            j=0;
            for v_index in range(0,num_of_vectors,1):
                if permutation_index == 0:
                    X_new = self.permuted(vectors[v_index],theta)
                    new_vectors.append(X_new[:self.K])
                else:
                    X_new = self.permuted(vectors[v_index],theta)
                    new_vectors[v_index] = X_new[:self.K]

                # C[i] = np.argsort(X_new[:self.K])[-self.m:].tolist()
                # C[i] = max(range(len(X_new[:self.K])), key=X_new[:self.K].__getitem__)
                # C[i*num_of_vectors + j] = max(range(len(X_new[:self.K])), key=X_new[:self.K].__getitem__)
                C[i*num_of_vectors + j] = max(range(len(X_new[:self.K])), key=X_new[:self.K].__getitem__)

                # if permutation_index % 2 == 0: 
                #     C[i*num_of_vectors + j] = max(range(len(X_new[:self.K])), key=X_new[:self.K].__getitem__)
                # else:
                #     C[i*num_of_vectors + j] = min(range(len(X_new[-self.K:])), key=X_new[-self.K:].__getitem__)
                    
                self.bucket_insert(buckets, str(str(C[i*num_of_vectors + j])+"-"+str(i)), v_index)
                j+=1;
            i+=1;


        return buckets, np.array(new_vectors,dtype=np.intp)

    def permuted(self, vector, permutation):
        permuted_vector = [vector[x] for x in permutation]
        return permuted_vector

    def bucket_insert(self, buckets, hashCode, value):
        if hashCode not in buckets.keys():
            buckets[hashCode] = set()
        buckets[hashCode].add(value)
        return buckets
    
def wta_similarity(vector1, vector2):

    PO=0
    for i in range(0,len(vector1),1):
        for j in range(0,i,1):
            ij_1 = vector1[i] - vector1[j]
            ij_2 = vector2[i] - vector2[j]
            PO += wta_threshold(ij_1*ij_2)
            
    return PO

def wta_threshold(x):    
    
    if x>0:
        return 1
    else:
        return 0
