# Import Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
# Download dataset dari: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
df = pd.read_csv("Mall_Customers.csv")

# Lihat lima baris pertama
print(df.head())

# 2. Pemrosesan Data
# Rename kolom untuk mempermudah
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']

# Konversi Gender menjadi numerik
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Pilih fitur untuk clustering
features = df[['AnnualIncome', 'SpendingScore']]

# Standarisasi data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Tentukan Jumlah Cluster Optimal (Elbow Method)
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 4. K-Means Clustering
# Dari Elbow Method, pilih jumlah cluster optimal (misalnya, 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# 5. Visualisasi Cluster
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=df['AnnualIncome'], y=df['SpendingScore'], 
    hue=df['Cluster'], palette='viridis', s=100
)
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# 6. Analisis Deskriptif untuk Setiap Cluster
# Tampilkan statistik rata-rata untuk setiap cluster
cluster_analysis = df.groupby('Cluster').mean()
print("Cluster Analysis:\n", cluster_analysis)

# Simpan hasil clustering ke file baru
df.to_csv("customer_segmentation_results.csv", index=False)
print("Hasil clustering disimpan ke 'customer_segmentation_results.csv'")
