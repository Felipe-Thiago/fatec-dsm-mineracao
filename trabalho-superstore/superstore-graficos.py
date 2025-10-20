# Arquivo de clusterização e gráficos dos dados normalizados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist

from mpl_toolkits.mplot3d import Axes3D

# Leitura do arquivo CSV gerado após normalização
df = pd.read_csv("relatorio_superstore_tratado3.csv", encoding='utf-8')

# Seleção dos dados numéricos para clusterização (usados como base)
X = df[["Sales", "Discount", "Profit"]]


# Aplicação do K-Means para clusterização
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_
print("Centroides dos clusters:\n", kmeans.cluster_centers_)
print("Número de clusters formados:", len(set(kmeans.labels_)))

# Cálculo do silhouette score
silhouette = silhouette_score(X, df['Cluster'])
print(f"\nSilhouette Score: {silhouette:.4f}")

# Cálculo do índice davies-bouldin
davies_bouldin = davies_bouldin_score(X, df['Cluster'])
print(f"Índice Davies-Bouldin: {davies_bouldin:.4f}")

# Cálculo do índice Dunn
def dunn_index(X, labels):
    unique_clusters = set(labels)
    inter_cluster_distances = []
    intra_cluster_distances = []

    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        if len(cluster_points) > 1:
            intra_dist = np.mean(cdist(cluster_points, cluster_points, 'euclidean'))
            intra_cluster_distances.append(intra_dist)

    for i in unique_clusters:
        for j in unique_clusters:
            if i < j:
                cluster_i = X[labels == i]
                cluster_j = X[labels == j]
                inter_dist = np.min(cdist(cluster_i, cluster_j, 'euclidean'))
                inter_cluster_distances.append(inter_dist)

    dunn_index_value = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
    return dunn_index_value

dunn = dunn_index(X.values, df['Cluster'])
print(f"Índice Dunn: {dunn:.4f}")



# Visualização dos clusters em um gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Sales'], df['Discount'], df['Profit'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Sales')
ax.set_ylabel('Discount')
ax.set_zlabel('Profit')
plt.title('Clusters de Produtos')
plt.show()

# Gráficos dos atributos numéricos e categóricos para cada cluster
plt.figure(figsize=(20, 15))
for cluster in df['Cluster'].unique():
    plt.subplot(2, 2, cluster + 1)
    cluster_data = df[df['Cluster'] == cluster]
    sns.histplot(cluster_data['Sales'], kde=True)
    plt.title(f"Cluster {cluster} - Sales")
    plt.xlabel("Sales")
    plt.ylabel("Density")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
for cluster in df['Cluster'].unique():
    plt.subplot(2, 2, cluster + 1)
    cluster_data = df[df['Cluster'] == cluster]
    sns.histplot(cluster_data['Discount'], kde=True)
    plt.title(f"Cluster {cluster} - Discount")
    plt.xlabel("Discount")
    plt.ylabel("Density")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
for cluster in df['Cluster'].unique():
    plt.subplot(2, 2, cluster + 1)
    cluster_data = df[df['Cluster'] == cluster]
    sns.histplot(cluster_data['Profit'], kde=True)
    plt.title(f"Cluster {cluster} - Profit")
    plt.xlabel("Profit")
    plt.ylabel("Density")
plt.tight_layout()
plt.show()

# # Gráficos categóricos para cada cluster
colunas = ["Segment", "Country", "State", "Category", "Sub-Category"]

for col in colunas:
    plt.figure(figsize=(20, 15))
    for cluster in df['Cluster'].unique():
        plt.subplot(2, 2, cluster + 1)
        cluster_data = df[df['Cluster'] == cluster]
        sns.countplot(data=cluster_data, x=col)
        plt.title(f"Cluster {cluster} - Contagem de {col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Exportando o DataFrame com clusters para um arquivo CSV
df.to_csv("relatorio_superstore_com_clusters2.csv", index=False, encoding='utf-8')
print("Relatório gerado: relatorio_superstore_com_clusters2.csv")