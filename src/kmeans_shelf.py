from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Load and preprocess data
df = pd.read_csv("Static_Training_Dataset_2.csv", sep=',')
df['Time_start'] = pd.to_datetime(df['Time_start'])
df['Time_finish'] = pd.to_datetime(df['Time_finish'])
df['Time_spent'] = (df['Time_finish'] - df['Time_start']).dt.total_seconds()  # Convert to seconds

# Invert Time_spent (lower values = faster movement = higher priority)
df['Time_priority'] = 1 / (1 + df['Time_spent'])  # Adding 1 to avoid division by zero

# Create frequency count per brand
brand_counts = df['Company_Brand'].value_counts().reset_index()
brand_counts.columns = ['Company_Brand', 'Count']
df = pd.merge(df, brand_counts, on='Company_Brand')

# Select features for K-means
kmeans_df = df[['Income', 'Time_priority', 'Count']].copy()

# Standardize features (critical for K-means)
scaler = StandardScaler()
kmeans_data = scaler.fit_transform(kmeans_df)

dataframe = pd.read_csv("GUI_part4_distance.csv") 
Total_shelves = dataframe['ID_Total'].max()    

# Apply K-means
kmeans = KMeans(n_clusters=Total_shelves, random_state=42)
df['Shelf_Cluster'] = kmeans.fit_predict(kmeans_data) + 1

# Analyze cluster characteristics
cluster_summary = df.groupby('Shelf_Cluster')[['Income', 'Time_priority', 'Count']].mean()
print(cluster_summary)

# Save results
df.to_csv('KMeans_shelf_locations.csv', index=False)

# Create a brand-cluster lookup table
brand_cluster_map = df.groupby('Company_Brand')['Shelf_Cluster'].agg(lambda x: x.mode()[0]).to_dict()

print(brand_cluster_map)