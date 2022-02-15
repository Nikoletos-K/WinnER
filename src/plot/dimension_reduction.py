import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from tqdm.notebook import tqdm as tqdm
from sklearn.decomposition import PCA 
from sklearn.manifold import MDS
import os

def SpaceVisualization2D(X, prototypes, withPlotly=True, withText=False, decompositionMenthod='PCA'):

    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=2)
    else:
        decompositionComponents = MDS(n_components=2)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component = components[:, 0], components[:, 1]    

    if withPlotly:
        prototypes = [ '1' if i in prototypes else '0' for i in range(0,len(first_component),1)]
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "prototypes": prototypes})
        fig = px.scatter(df, x="first_component", y="second_component", color="prototypes",symbol='prototypes',opacity=0.8, template='plotly_white',symbol_sequence= ['circle','circle-open'],color_discrete_sequence = ['darkblue', 'lightblue'],
            labels={
                     "first_component": "First component",
                     "second_component": "Second component",
                     "labels": "is Prototype"
            }
        )        
        fig.update_layout(
            title = {
                'text':  "2D Space Visualization with " + decompositionMenthod + " - Prototype Selection ",
                'y':0.9, 'x':0.5,
                'xanchor': 'center', 'yanchor': 'top'
            }
        )
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
        prototypes = [ '1' if i in prototypes else '0' for i in range(0,len(first_component),1)]
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "third_component": third_component, "prototypes": prototypes})
        fig = px.scatter_3d(df, x='first_component', y='second_component', z='third_component',  color='prototypes', opacity=0.6, template='plotly_white',symbol_sequence= ['circle','circle-open'],color_discrete_sequence = ['darkblue', 'lightblue'],
            labels={
                     "first_component": "First component",
                     "second_component": "Second component",
                     "second_component": "Third component",
                     "labels": "is Prototype"
            }
        )        
        fig.update_layout(
            title = {
                'text':  "3D Space Visualization with " + decompositionMenthod + " - Prototype Selection ",
                'y':0.9, 'x':0.5,
                'xanchor': 'center', 'yanchor': 'top'
            }, margin=dict(l=0, r=0, b=0, t=0)
        )
        fig.update_traces(marker=dict(size=3))
        fig.show()
        fig.write_image("SpaceVisualization3D_"+ decompositionMenthod +".png")
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
        
def SpaceVisualizationEmbeddings2D(X, labels, withPlotly=True, decompositionMenthod='PCA'):

    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=2)
    else:
        decompositionComponents = MDS(n_components=2)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component = components[:, 0], components[:, 1]

    if withPlotly:
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "labels": labels})
        fig = px.scatter(df, x="first_component", y="second_component", color="labels", opacity=0.8, template='plotly_white',
                labels={
                     "first_component": "First component",
                     "second_component": "Second component",
                     "labels": "Groups"
                 }
        )        
        fig.update_layout(
            title = {
                'text':  "2D Space Visualization with " + decompositionMenthod + " from the Embeddings",
                'y':0.9, 'x':0.5,
                'xanchor': 'center', 'yanchor': 'top'
            }
        )
        fig.show()
        fig.write_image("SpaceVisualizationEmbeddings2D_"+ decompositionMenthod +".png")
    else:
        title= '2D Space Visualization with ' + decompositionMenthod
        fig, ax = plt.subplots(figsize=(12,10))
        cm = plt.get_cmap('jet') 
        ax.scatter(first_component, second_component, c = labels, cmap=cm, s=30) 
        fig.suptitle(title,fontsize=15,fontweight="bold")
        plt.show()

def SpaceVisualizationEmbeddings3D(X, labels, withPlotly=True, decompositionMenthod='PCA'):
    
    if decompositionMenthod == 'PCA':
        decompositionComponents = PCA(n_components=3)
    else:
        decompositionComponents = MDS(n_components=3)
    decompositionComponents.fit(X)
    components = decompositionComponents.fit_transform(X)
    first_component, second_component, third_component = components[:, 0], components[:, 1], components[:, 2]

    if withPlotly:
        df = pd.DataFrame({"first_component": first_component, "second_component": second_component, "third_component": third_component, "labels": labels})
        fig = px.scatter_3d(df, x='first_component', y='second_component', z='third_component',  color='labels', opacity=0.7, template='plotly_white',
            labels={
                     "first_component": "First component",
                     "second_component": "Second component",
                     "second_component": "Third component",
                     "labels": "Groups"
            }
        )        
        fig.update_layout(
            title = {
                'text':  "3D Space Visualization with " + decompositionMenthod  + " from the Embeddings",
                'y':0.9, 'x':0.5,
                'xanchor': 'center', 'yanchor': 'top'
            }, margin=dict(l=0, r=0, b=0, t=0)
        )
        fig.update_traces(marker=dict(size=4))
        fig.show()
        fig.write_image("SpaceVisualizationEmbeddings3D_"+ decompositionMenthod +".png")
    else:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.get_cmap('jet') 
        title='3D Space Visualization from Embeddings with ' + decompositionMenthod

        if decompositionMenthod == 'PCA':
            print("Explained varianse of PCA:", decompositionComponents.explained_variance_ratio_)
        ax.scatter(first_component, second_component, third_component, c = labels, cmap=cm, s=30) 
        fig.suptitle(title,fontsize=15,fontweight="bold")
        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")
        plt.show()
