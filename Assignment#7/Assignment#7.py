import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans

data = load_wine()
X = data.data
y = data.target

df = pd.DataFrame(data.data, columns=data.feature_names)
print("Available columns:")
print(data.feature_names)

# 1. Choose two columns from the wine dataset to use for clustering.
feature1 = 'alcohol'
feature2 = 'flavanoids'
X_subset = df[[feature1, feature2]]

plt.figure(figsize=(14, 6))

# 2. Create a scatterplot of the two columns you selected.
plt.subplot(1, 2, 1) 
plt.scatter(df[feature1], df[feature2], color='gray', edgecolors='black', alpha=0.7)
plt.title(f"Original Data: {feature1} vs {feature2}")
plt.xlabel(feature1)
plt.ylabel(feature2)

# 3. Run k-means clustering with n_clusters=3 using only your two selected columns.
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_subset)

# 4. Add the cluster labels predicted by k-means to your DataFrame.
df['Cluster'] = kmeans.labels_
print("\nFirst 5 rows of DataFrame with new Cluster labels:")
print(df[[feature1, feature2, 'Cluster']].head())

# 5. Create a second scatterplot of the same two columns, colored by k-means cluster.
plt.subplot(1, 2, 2)
plt.scatter(df[feature1], df[feature2], c=df['Cluster'], cmap='viridis', edgecolors='black', alpha=0.8)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

plt.title("K-Means Clustering (n=3)")
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

plt.tight_layout()
plt.show()