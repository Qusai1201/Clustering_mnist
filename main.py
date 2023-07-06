import keras
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')


X_resized = np.resize(X, (X.shape[0], 8, 8))
X = X_resized.reshape(X_resized.shape[0], -1)

Clus_dataSet = StandardScaler().fit_transform(X)

variance = 0.40
pca = PCA(variance)

pca.fit(Clus_dataSet)

Clus_dataSet = pca.transform(Clus_dataSet)

k_means = KMeans(init = "k-means++", n_clusters = 10, n_init = 35)
k_means.fit(Clus_dataSet)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_


silhouette_score = silhouette_score(Clus_dataSet, k_means_labels)
print("silhouette_score : " , silhouette_score)

import plotly as py
import plotly.graph_objs as go
import plotly.express as px




layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)

colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
trace = [ go.Scatter3d() for _ in range(11)]
for i in range(0,10):
    my_members = (k_means_labels == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
            x=Clus_dataSet[my_members, 0],
            y=Clus_dataSet[my_members, 1],
            z=Clus_dataSet[my_members, 2],
            mode='markers',
            marker = dict(size = 2,color = colors[i]),
            hovertext=index,
            name='Cluster'+str(i),
            )

fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]], layout=layout)
    
py.offline.iplot(fig)
