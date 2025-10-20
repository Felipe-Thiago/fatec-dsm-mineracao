# Importação das bibliotecas

# python -m pip install pandas 
# python -m pip install matplotlib
# python -m pip install seaborn
# python -m pip install scikit-learn
# python -m pip install scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import PowerTransformer, RobustScaler

# Download do dataset na nuvem kaggle
# path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
# print("Path to dataset files:", path)
'''
Observação: Arquivo kaggle utilizado em forma de download manual, encontrado através do link: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final?resource=download
'''

# Leitura do arquivo CSV
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

# Seleção dos dados
dados = df[["Product Name", "Segment", "Country", "State", "Category", "Sub-Category", "Sales", "Discount", "Profit"]]
num_cols = ["Sales", "Discount", "Profit"]

# ------------------------ normalização ------------------------

# Inspecionar assimetria (skew) antes
print("Assimetria antes:\n", dados[num_cols].skew())



# Power Transformer (Yeo-Johnson) para atributos numéricos
pt = PowerTransformer(method='yeo-johnson', standardize=True)
dados[num_cols] = pt.fit_transform(dados[num_cols])

# Normalização dos dados numéricos usando robust scaling com 6 casas decimais ------------------------
scaler = RobustScaler()
dados[num_cols] = scaler.fit_transform(dados[num_cols]).round(6)

# Distribuição dos valores em 5 bins de mesma largura (Equi-width)
dados['Sales_bin'] = pd.cut(dados['Sales'], bins=5, labels=False)
dados['Discount_bin'] = pd.cut(dados['Discount'], bins=5, labels=False)
dados['Profit_bin'] = pd.cut(dados['Profit'], bins=5, labels=False)



# Inspecionar assimetria depois
print("Assimetria depois:\n", dados[num_cols].skew())


# ---------------- outliers -------------------------


# Identificando outliers em 'Profit'
Q1_profit = dados['Profit'].quantile(0.25)
Q3_profit = dados['Profit'].quantile(0.75)
IQR = Q3_profit - Q1_profit
outliers_profit = dados[(dados['Profit'] < Q1_profit - 1.5 * IQR) | (dados['Profit'] > Q3_profit + 1.5 * IQR)]
print("Outliers em Profit:\n", outliers_profit[['Product Name', 'Profit']])

# Identificando outliers em 'Sales'
Q1_sales = dados['Sales'].quantile(0.25)
Q3_sales = dados['Sales'].quantile(0.75)
IQR_sales = Q3_sales - Q1_sales
outliers_sales = dados[(dados['Sales'] < Q1_sales - 1.5 * IQR_sales) | (dados['Sales'] > Q3_sales + 1.5 * IQR_sales)]
print("Outliers em Sales:\n", outliers_sales[['Product Name', 'Sales']])

# Identificando outliers em 'Discount'
Q1_discount = dados['Discount'].quantile(0.25)
Q3_discount = dados['Discount'].quantile(0.75)
IQR_discount = Q3_discount - Q1_discount
outliers_discount = dados[(dados['Discount'] < Q1_discount - 1.5 * IQR_discount) | (dados['Discount'] > Q3_discount + 1.5 * IQR_discount)]
print("Outliers em Discount:\n", outliers_discount[['Product Name', 'Discount']])

# Retirando outliers do DataFrame
dados = dados[~dados.index.isin(outliers_profit.index)]
dados = dados[~dados.index.isin(outliers_sales.index)]
dados = dados[~dados.index.isin(outliers_discount.index)]

# Retirando as colunas de bins
dados = dados.drop(columns=['Sales_bin', 'Discount_bin', 'Profit_bin'])


# --------------- gráficos -------------------------


# Gráficos dos atributos numéricos e categóricos
# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=dados[["Sales", "Discount", "Profit"]])
plt.title("Boxplot dos atributos numéricos")
plt.show()



# plt.figure(figsize=(20, 15))

# # Histogramas
# plt.subplot(2, 4, 1)
# dados["Sales"].hist()
# plt.title("Histograma Sales")

# plt.subplot(2, 4, 2)
# dados["Discount"].hist()
# plt.title("Histograma Discount")

# plt.subplot(2, 4, 3)
# dados["Profit"].hist()
# plt.title("Histograma Profit")

# Gráficos categóricos
# colunas = ["Segment", "Country", "State", "Category", "Sub-Category"]
# for i, col in enumerate(colunas):
#     plt.subplot(2, 4, i+4)
#     sns.countplot(data=dados, x=col)
#     plt.title(f"Contagem de {col}")
#     plt.xticks(rotation=45)

# # Ajustar o layout
# plt.tight_layout()
# plt.show()


# --------------- exportação -------------------------


# Exportando o novo DataFrame para um arquivo CSV
dados.to_csv("relatorio_superstore_tratado3.csv", index=False, encoding='utf-8')
print("Relatório gerado: relatorio_superstore_tratado3.csv")