# Arquivo de clusterização e gráficos dos dados normalizados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Leitura do arquivo CSV gerado após normalização e remoção de outliers
df = pd.read_csv("relatorio_superstore_tratado2.csv", encoding='utf-8')

# Seleção dos dados numéricos para clusterização
X = df[["Sales", "Discount", "Profit"]]

# Aplicação do K-Means para clusterização
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_
# Visualização dos clusters em um gráfico 3D
from mpl_toolkits.mplot3d import Axes3D
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
df.to_csv("relatorio_superstore_com_clusters.csv", index=False, encoding='utf-8')
print("Relatório gerado: relatorio_superstore_com_clusters.csv")