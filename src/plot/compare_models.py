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

scores_dataframe = pd.read_csv("../results/cora/scores.csv")

fig = px.line(scores_dataframe, x = "Recall",  y="Workflow", color='Workflow')
fig.show()