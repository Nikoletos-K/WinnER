import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from tqdm.notebook import tqdm as tqdm
from sklearn.decomposition import PCA 
from sklearn.manifold import MDS

def SpaceVisualization2D(X, prototypes, withPlotly=True, withText=False, decompositionMenthod='PCA'):

    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=2)
    else:
        decompositionComponents = MDS(n_components=2)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component = components[:, 0], components[:, 1]    

    if withPlotly:
        prototypes = [ 1 if i in prototypes else 0 for i in range(0,len(first_component),1)]
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "prototypes": prototypes})
        fig = px.scatter(df, x="first_component", y="second_component", color="prototypes",symbol='prototypes', template='plotly_white')
        fig.update_traces(marker_size=8, marker_coloraxis=None)
        fig.show()
    else:
        fig, _ = plt.subplots(figsize=(14,8))
        title='2D Space Visualization with ' + decompositionMenthod
        fig.suptitle(title,fontsize=10,fontweight="bold")
        
        for x0, y0, i in zip(first_component, second_component, range(0,len(first_component),1)):
            if(withText):
                if i in prototypes:
                    plt.text(x0, y0, i, ha="center", va="center", fontsize=16, color='r', fontweight="bold")
                else:
                    plt.text(x0, y0, i, ha="center", va="center", fontsize=8, color='b')
            else:
                if i in prototypes:
                    plt.scatter(x0, y0, color='r', s=250, marker='*', alpha=1.0)
                else:
                    plt.scatter(x0, y0, color='b', s=80, marker='.', alpha=1.0)
        plt.show()

def SpaceVisualization3D(X, prototypes, withText=False, withPlotly=True, decompositionMenthod='PCA'):
    
    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=3)
    else:
        decompositionComponents = MDS(n_components=3)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    
    first_component, second_component, third_component = components[:, 0], components[:, 1], components[:, 2] 
    
    if withPlotly:
        prototypes = [ 1 if i in prototypes else 0 for i in range(0,len(first_component),1)]
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "third_component": third_component, "prototypes": prototypes})
        fig = px.scatter_3d(df, x='first_component', y='second_component', z='third_component',  color='prototypes', size_max=2, opacity=0.8, template='plotly_white')
        # fig = go.Figure(data=[go.Scatter3d(
        #     x=first_component,
        #     y=second_component,
        #     z=third_component,
        #     mode='markers',
        #     marker=dict(
        #         size=8,
        #         color=third_component,                # set color to an array/list of desired values
        #         colorscale='Viridis',   # choose a colorscale
        #         opacity=0.8
        #     )
        # )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
    else:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        for x0, y0, z0, i in zip(first_component, second_component, third_component, range(0,len(first_component),1)):
            if(withText):
                if i in prototypes:
                    plt.text(x0, y0, z0, i, ha="center", va="center", fontsize=16, color='r', fontweight="bold")
                else:
                    plt.text(x0, y0, z0, i, ha="center", va="center", fontsize=8, color='b')
            else:
                if i in prototypes:
                    ax.scatter(x0, y0, z0, color='r', s=250, marker='*', alpha=1.0)
                else:
                    ax.scatter(x0, y0, z0, color='b', s=80, marker='.', alpha=1.0)
                    
        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")

        plt.show()
        
def SpaceVisualizationEmbeddings2D(X, prototypes, labels, withPlotly=True, withgroundruth=False, groundruth=None, decompositionMenthod='PCA'):

    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=2)
    else:
        decompositionComponents = MDS(n_components=2)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component = components[:, 0], components[:, 1]

    if withPlotly:
        prototypes = [ 1 if i in prototypes else 0 for i in range(0,len(first_component),1)]
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "prototypes": prototypes})
        fig = px.scatter(df, x="first_component", y="second_component", color="prototypes",symbol='prototypes',color_discrete_map={'1': 'lightcyan', '0': 'darkblue'}, template='plotly_white')
        fig.update_traces(marker_size=8, marker_coloraxis=None)
        fig.show()
    else:
        title= '2D Space Visualization with ' + decompositionMenthod
        fig, ax = plt.subplots(figsize=(12,10))
        cm = plt.get_cmap('jet') 
        
        if not withgroundruth:
            labels = [",".join(item) for item in labels.astype(str)]
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
                if i in set(prototypes):
                    plt.scatter(x0,y0,c='red',s=400,marker='*',alpha=0.2)
        plt.show()

def SpaceVisualizationEmbeddings3D(X, prototypes, labels, withPlotly=True, withgroundruth=False, groundruth=None, decompositionMenthod='PCA'):
    
    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=3)
    else:
        decompositionComponents = MDS(n_components=3)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component, third_component = components[:, 0], components[:, 1], components[:, 2]

    if withPlotly:
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "third_component": third_component, "labels": groundruth})
        fig = px.scatter_3d(df, x='first_component', y='second_component', z='third_component',  color='labels', size_max=2, opacity=0.8, template='plotly_white')
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
    else:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.get_cmap('jet') 
        title='3D Space Visualization from Embeddings with ' + decompositionMenthod
        
        if decompositionMenthod == 'PCA':
            print("Explained varianse of PCA:", decompositionComponents.explained_variance_ratio_)

        if not withgroundruth:
            labels = [",".join(item) for item in labels.astype(str)]
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
                if i in set(prototypes):
                    ax.scatter(x0,y0,z0,c='red',s=400,marker='*',alpha=0.2)
                    
        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")

        plt.show()