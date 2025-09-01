
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


data = pd.read_csv("../../datasets/iris.csv") 
X = data.iloc[:, :-1].values  


Z = linkage(X, method='ward') 


plt.figure(figsize=(10, 5))
dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')  
labels = cluster_model.fit_predict(X)


data['Cluster'] = labels
print(data.head())
