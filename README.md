import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# ======================= STEP 1: Dataset Collection & Understanding =======================
file_path = 'online_retail.csv'  # change if needed
df = pd.read_csv(file_path, encoding='ISO-8859-1')

print("\n=== Dataset Info ===")
df.info()
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

neg_qty = df[df['Quantity'] <= 0].shape[0]
neg_price = df[df['UnitPrice'] <= 0].shape[0]
print(f"\nRecords with Quantity <= 0: {neg_qty}")
print(f"Records with UnitPrice <= 0: {neg_price}")

# ======================= STEP 2: Data Preprocessing =======================
print("\n=== Data Cleaning ===")
df_clean = df.dropna(subset=['CustomerID'])
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
df_clean = df_clean.dropna(subset=['InvoiceDate'])
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

df_clean.to_csv('cleaned_online_retail.csv', index=False)
print("Cleaned data saved to cleaned_online_retail.csv")

# ======================= STEP 3: Exploratory Data Analysis (EDA) =======================
# Transaction Volume by Country
plt.figure(figsize=(12,6))
country_sales = df_clean.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
sns.barplot(x=country_sales.values, y=country_sales.index)
plt.title("Transaction Volume by Country")
plt.xlabel("Number of Transactions")
plt.ylabel("Country")
plt.show()

# Top-Selling Products
plt.figure(figsize=(12,6))
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.show()

# Purchase Trends Over Time
df_clean.set_index('InvoiceDate', inplace=True)
monthly_sales = df_clean.resample('M')['TotalPrice'].sum()
plt.figure(figsize=(12,6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.ylabel("Total Sales (£)")
plt.xlabel("Date")
plt.show()
df_clean.reset_index(inplace=True)

# Monetary Distribution per Transaction
transaction_amounts = df_clean.groupby('InvoiceNo')['TotalPrice'].sum()
plt.figure(figsize=(10,6))
sns.histplot(transaction_amounts, bins=50, kde=True)
plt.title("Monetary Value per Transaction")
plt.xlabel("Total (£)")
plt.show()

# Monetary Distribution per Customer
customer_amounts = df_clean.groupby('CustomerID')['TotalPrice'].sum()
plt.figure(figsize=(10,6))
sns.histplot(customer_amounts, bins=50, kde=True)
plt.title("Monetary Value per Customer")
plt.xlabel("Total (£)")
plt.show()

# ======================= STEP 4: Clustering Methodology =======================
print("\n=== RFM Feature Engineering ===")
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
})

# Standardize RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Elbow & Silhouette
sse, silhouette_scores = [], []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

plt.figure(figsize=(8,5))
plt.plot(K_range, sse, marker='o')
plt.title("Elbow Curve for RFM Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score for RFM Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Final Clustering
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, n_init=10, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualize Clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='tab10')
plt.title("Customer Segments")
plt.show()

# Save model & scaler
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Clustering model and scaler saved.")

# Label Clusters
cluster_summary = rfm.groupby('Cluster').mean().round(2)
print("\nCluster Profiles:")
print(cluster_summary)

labels_map = {}
for idx, row in cluster_summary.iterrows():
    if row['Recency'] < rfm['Recency'].quantile(0.25) and row['Frequency'] > rfm['Frequency'].quantile(0.75) and row['Monetary'] > rfm['Monetary'].quantile(0.75):
        labels_map[idx] = 'High-Value'
    elif row['Frequency'] > rfm['Frequency'].median() and row['Monetary'] > rfm['Monetary'].median():
        labels_map[idx] = 'Regular'
    elif row['Frequency'] < rfm['Frequency'].median() and row['Monetary'] < rfm['Monetary'].median():
        labels_map[idx] = 'Occasional'
    else:
        labels_map[idx] = 'At-Risk'

rfm['Segment'] = rfm['Cluster'].map(labels_map)

print("\nCluster Segment Labels Assigned:")
print(rfm[['Cluster', 'Segment']].head())

rfm.to_csv("rfm_with_clusters.csv")
print("RFM table with clusters & segments saved to rfm_with_clusters.csv")

# ======================= STEP 5: Recommendation System =======================
print("\n=== Recommendation System ===")

# Customer-Product Matrix
basket = df_clean.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)

# Cosine Similarity
product_similarity = pd.DataFrame(cosine_similarity(basket.T), 
                                   index=basket.columns, columns=basket.columns)

with open("product_similarity.pkl", "wb") as f:
    pickle.dump(product_similarity, f)

print("Product similarity matrix saved to product_similarity.pkl")

def recommend_products(product_name, n=5):
    if product_name not in product_similarity.columns:
        print(f"Product '{product_name}' not found.")
        return []
    similar_products = product_similarity[product_name].sort_values(ascending=False).drop(product_name).head(n)
    return similar_products

# Example usage
example_product = basket.columns[0]
print(f"\nTop 5 products similar to '{example_product}':")
print(recommend_products(example_product))
