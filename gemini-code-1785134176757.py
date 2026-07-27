import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Setup & Config
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(42)

# 2. Generate Synthetic Dataset (Replace with pd.read_csv("your_data.csv") if needed)
n_records = 5000
dates = pd.date_range(start="2025-01-01", end="2026-06-30", freq="D")

data = {
    "CustomerID": np.random.randint(10000, 10500, size=n_records),
    "InvoiceDate": np.random.choice(dates, size=n_records),
    "TotalAmount": np.random.exponential(scale=80, size=n_records).round(2) + 5
}
df = pd.DataFrame(data)

# 3. RFM Feature Extraction
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'CustomerID': 'count',
    'TotalAmount': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'CustomerID': 'Frequency',
    'TotalAmount': 'Monetary'
})

# 4. Data Transformation & Scaling
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# 5. Model Evaluation (Elbow & Silhouette)
k_range = range(2, 9)
inertia, silhouette_scores = [], []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Plot Elbow & Silhouette
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color='tab:blue')
ax1.plot(k_range, inertia, marker='o', color='tab:blue')

ax2 = ax1.twinx()  
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.plot(k_range, silhouette_scores, marker='s', linestyle='--', color='tab:red')
plt.title('Optimal K Evaluation')
plt.show()

# 6. Fit Final Model (k=4) & Map Segments
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

segment_map = {
    0: "At-Risk Customers",
    1: "Champions / VIPs",
    2: "New / Recent Customers",
    3: "Loyal Customers"
}
rfm['Segment'] = rfm['Cluster'].map(segment_map)

# 7. Print Results & Visualisation
print("\n--- Segment Profile Summary ---")
summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)
print(summary)

# Plot Final Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rfm, 
    x='Frequency', 
    y='Monetary', 
    hue='Segment', 
    palette='Set2', 
    s=70
)
plt.yscale('log')
plt.title('Customer Segments: Frequency vs. Monetary Value')
plt.show()

# Save final results
rfm.to_csv("customer_segments_result.csv")
print("\nSuccess: Process completed and saved to 'customer_segments_result.csv'")