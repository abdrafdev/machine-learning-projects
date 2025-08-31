import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("../../datasets/iris.csv")
print("Columns in dataset:", data.columns)


X = data.iloc[:, :-1]


kmeans = KMeans(n_clusters=3, random_state=42)


labels = kmeans.fit_predict(X)


data["Cluster"] = labels
print("\nFirst 5 rows with cluster assignment:")
print(data.head())


centroids = kmeans.cluster_centers_


plt.figure(figsize=(10,6))


colors = ["red", "green", "blue"]
for i in range(3):
    plt.scatter(
        X.iloc[labels == i, 0],  
        X.iloc[labels == i, 1],  
        s=50,
        c=colors[i],
        label=f'Cluster {i}'
    )


plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=200,
    c='yellow',
    marker='X',
    edgecolor='black',
    label='Centroids'
)


plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("K-Means Clustering on Iris Dataset (with Centroids)")
plt.legend()
plt.grid(True)
plt.show()
