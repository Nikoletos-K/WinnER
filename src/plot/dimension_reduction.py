import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm as tqdm
from sklearn.decomposition import PCA 
from sklearn.manifold import MDS

def PCA_SpaceVisualization(X,Prototypes,title='PCA visualization',withText=False,decompositionMenthod='PCA'):
    '''
    PCA to given array X and creating a plot
    Returns PCA components array after fit_transform
    '''
    
    # PCA code
        # PCA code
    if decompositionMenthod == 'PCA':
        pca = PCA(n_components=2)
    else:
        pca = MDS(n_components=2)
    pca.fit(X)
    pcaComponents = pca.fit_transform(X) # pcaComponents is the data that I'll use from PCA
    

    # Seperating components
    first_component = [x[0] for x in pcaComponents]
    second_component = [x[1] for x in pcaComponents]
    
    # Plotting code
    fig, ax = plt.subplots(figsize=(10,8))
    fig.suptitle(title,fontsize=15,fontweight="bold")
    
    for x0, y0, i in zip(first_component, second_component,range(0,len(first_component),1)):
        if(withText):
            if i in Prototypes:
                plt.text(x0,y0,i, ha="center", va="center",fontsize=16,color='r',fontweight="bold")
            else:
                plt.text(x0,y0,i, ha="center", va="center",fontsize=8,color='b')
        else:
            if i in Prototypes:
                plt.scatter(x0,y0,color='r',s=250,marker='*',alpha=1.0)
            else:
                plt.scatter(x0,y0,color='b',s=80,marker='.',alpha=1.0)
    
    plt.show()

    return pcaComponents

def PCA_SpaceVisualization_3D(X,Prototypes,title='PCA visualization',withText=False,decompositionMenthod='PCA'):
    
    # PCA code
    if decompositionMenthod == 'PCA':
        pca = PCA(n_components=3)
    else:
        pca = MDS(n_components=3)
    pca.fit(X)
    pcaComponents = pca.fit_transform(X) # pcaComponents is the data that I'll use from PCA
    
    # Seperating components
    first_component = [x[0] for x in pcaComponents]
    second_component = [x[1] for x in pcaComponents]
    third_component = [x[2] for x in pcaComponents]

    # Plotting code
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    if decompositionMenthod == 'PCA':
        print("Explained varianse of PCA:", pca.explained_variance_ratio_)

    
    for x0, y0, z0, i in zip(first_component, second_component, third_component, range(0,len(first_component),1)):
        if(withText):
            if i in Prototypes:
                plt.text(x0, y0, z0, i, ha="center", va="center", fontsize=16, color='r', fontweight="bold")
            else:
                plt.text(x0, y0, z0, i, ha="center", va="center", fontsize=8, color='b')
        else:
            if i in Prototypes:
                ax.scatter(x0, y0, z0, color='r', s=250, marker='*', alpha=1.0)
            else:
                ax.scatter(x0, y0, z0, color='b', s=80, marker='.', alpha=1.0)
                
    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()
    
    return pcaComponents

def WTA_PCA_SpaceVisualization(X,Prototypes,Labels,title='PCA visualization',withText=False,withgroundruth=False,groundruth=None,decompositionMenthod='PCA'):
    '''
    PCA to given array X and creating a plot
    Returns PCA components array after fit_transform
    '''
    
    # PCA code
    if decompositionMenthod == 'PCA':
        pca = PCA(n_components=2)
    else:
        pca = MDS(n_components=2)
    pca.fit(X)
    pcaComponents = pca.fit_transform(X) # pcaComponents is the data that I'll use from PCA
    
    # Seperating components
    first_component = [x[0] for x in pcaComponents]
    second_component = [x[1] for x in pcaComponents]
    
    # Plotting code
    fig, ax = plt.subplots(figsize=(12,10))
    cm = plt.get_cmap('jet') 
    
    if not withgroundruth:
        labels = [",".join(item) for item in Labels.astype(str)]
        mydict={}
        i = 0
        for item in labels:
            if(i>0 and item in mydict):
                continue
            else:    
               i = i+1
               mydict[item] = i

        k=[]
        for item in labels:
            k.append(mydict[item])
        
    else:
        k=groundruth

    ax.scatter(first_component, second_component, c = k, cmap=cm, s=30) 
    fig.suptitle(title,fontsize=15,fontweight="bold")
    
    if not withgroundruth:
        for x0, y0, i in zip(first_component, second_component,range(0,len(first_component),1)):
            if i in set(Prototypes):
                plt.scatter(x0,y0,c='red',s=400,marker='*',alpha=0.2)
    plt.show()
    
    return pcaComponents

def WTA_PCA_SpaceVisualization_3D(X,Prototypes,Labels,title='PCA visualization',withText=False,withgroundruth=False,groundruth=None,decompositionMenthod='PCA'):
    
    # PCA code
    if decompositionMenthod == 'PCA':
        pca = PCA(n_components=3)
    else:
        pca = MDS(n_components=3)
    pca.fit(X)
    pcaComponents = pca.fit_transform(X) # pcaComponents is the data that I'll use from PCA
    
    # Seperating components
    first_component = [x[0] for x in pcaComponents]
    second_component = [x[1] for x in pcaComponents]
    third_component = [x[2] for x in pcaComponents]

    # Plotting code
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.get_cmap('jet') 

    if decompositionMenthod == 'PCA':
        print("Explained varianse of PCA:", pca.explained_variance_ratio_)

    if not withgroundruth:
        print(Labels)
        labels = [",".join(item) for item in Labels.astype(str)]
        mydict={}
        i = 0
        for item in labels:
            if(i>0 and item in mydict):
                continue
            else:    
               i = i+1
               mydict[item] = i
        k=[]
        for item in labels:
            k.append(mydict[item])
    else:
        k=groundruth

    ax.scatter(first_component, second_component, third_component, c = k, cmap=cm, s=30) 
    fig.suptitle(title,fontsize=15,fontweight="bold")
    
    if not withgroundruth:
        for x0, y0, z0, i in zip(first_component, second_component, third_component, range(0,len(first_component),1)):
            if i in set(Prototypes):
                ax.scatter(x0,y0,z0,c='red',s=400,marker='*',alpha=0.2)
                
    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()
    
    return pcaComponents