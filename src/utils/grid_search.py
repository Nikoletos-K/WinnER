import time
import pandas as pd
from tqdm.notebook import tqdm as tqdm
import numpy as np
from datetime import datetime


def GridSearch(rankedWTAHash, evaluate, model, data,true_matrix,max_numberOf_clusters,max_dissimilarityDistance,similarityThreshold,windowSize,metric,similarityVectors,distanceMetricEmbedding,distanceMetric,number_of_permutations,ngramms,withchars,prototypeFilter,earlyStop):
    results_dataframe = pd.DataFrame(columns=['max_numberOf_clusters','max_dissimilarityDistance','similarityThreshold','windowSize','metric','similarityVectors',"distanceMetricEmbedding","distanceMetric","number_of_permutations",'prototypesFilterThr',"protSelectionVariance",'numOfPrototypes','numOfBuckets','averageBucketSize','Accuracy','Precision','Recall','F1','Time'])
    i=1
    for n1 in tqdm(max_numberOf_clusters):
        for n2 in (max_dissimilarityDistance):
            for n3 in (similarityThreshold):
                for n4 in (windowSize):
                    for n5 in (metric):
                        for n6 in (similarityVectors):
                            for n7 in (distanceMetricEmbedding):
                                for n8 in (distanceMetric):
                                    for n9 in (number_of_permutations):
                                        for n10 in (withchars):
                                            for n11 in (withchars):
                                                for n12 in (prototypeFilter):
                                                    print("+ ------------  ",i,"   ------------- +")
                                                    print('max_numberOf_clusters: ',n1)
                                                    print('max_dissimilarityDistance: ',n2)
                                                    print('similarityThreshold: ',n3)
                                                    print('windowSize: ',n4)
                                                    print('metric: ',n5)
                                                    print('similarityVectors: ',n6)
                                                    print('distanceMetricEmbedding: ',n7)
                                                    print('distanceMetric: ',n8)
                                                    print('number_of_permutations: ',n9)
                                                    print('withchars: ',n10)
                                                    print('ngramms: ',n11)
                                                    print('prototypeFilter: ',n12)
                                                    print("+ ----------------------------------- +")
                                                    start = time.time()
                                                    model = rankedWTAHash(
                                                      earlyStop = earlyStop,
                                                      max_numberOf_clusters= n1,
                                                      max_dissimilarityDistance= n2,
                                                      windowSize= n4,
                                                      similarityThreshold= n3,
                                                      maxOnly= False,
                                                      metric=n5,
                                                      similarityVectors=n6,
                                                      number_of_permutations = n9,
                                                      distanceMetric= n8,
                                                      distanceMetricEmbedding = n7,
                                                      jaccard_withchars = n10,
                                                      ngramms= n11,                                                      
                                                      prototypesFilterThr = n12
                                                    )
                                                    model = model.fit(data)
                                                    exec_time = time.time() - start
                                                    if model.earlyStop==0:                                            
                                                        acc,f1,precision,recall = evaluate(model.mapping_matrix,true_matrix)
                                                    else:
                                                        if model.earlyStop == 3:
                                                            acc = f1 = precision = recall = 'Not counted'
                                                            averageBucketSize = np.mean([len(model.buckets[x]) for x in model.buckets.keys() ])
                                                            numOfBuckets=len(model.buckets.keys())
                                                        else:
                                                            numOfBuckets = averageBucketSize = acc = f1 = precision = recall = 'Not counted'
                                                    i+=1
                                                    results_dataframe.loc[len(results_dataframe)+1] = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n12,model.selectionVariance,model.selected_numOfPrototypes,numOfBuckets,averageBucketSize,acc,precision,recall,f1,exec_time]
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    results_dataframe.to_pickle(str(current_time)+".pkl")
    
    return results_dataframe