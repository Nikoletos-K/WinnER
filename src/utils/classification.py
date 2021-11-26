import numpy as np

def myMMD2(model):    
    mmdOnPrototypes = MMD2(model.dissimilarityDistance, model.S_index, model.prototypeArray)
    print("MMD2: ",mmdOnPrototypes)        
        
        
def MMD2(k, x, z):
    
    if type(x) is np.ndarray and type(z) is np.ndarray:
        m = z.size
        n = x.size
    else:
        m = len(z)
        n = len(x)
        
    firstSum = 0.0
    for i in range(0,m,1):
        for j in range(0,m,1):
            firstSum += k(z[i],z[j])
    
    secondSum = 0.0
    for i in range(0,m,1):
        for j in range(0,n,1):
            secondSum += k(z[i],x[j])
        
    thirdSum = 0.0
    for i in range(0,n,1):
        for j in range(0,n,1):
            thirdSum += k(x[i],x[j])
    
    mmd2 = (1/pow(m,2))*firstSum - (2/(m*n))*secondSum + (1/pow(n,2))*thirdSum
    
    
    return mmd2