import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import os


customer_data = pd.read_csv('data/shopping_data.csv')

data = customer_data.iloc[:,3:5].values

#print(data)

# plt.figure(figsize=(10, 7))
# plt.subplots_adjust(bottom=0.1)
# plt.scatter(data[:,0],data[:,1], label='True Position')

# plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
linked = linkage(data, 'single')
plt.figure(figsize=(10, 7))
dendrogram(linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True)
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()