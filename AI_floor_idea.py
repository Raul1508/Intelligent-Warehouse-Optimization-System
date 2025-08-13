level = int(Level.get())

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Load your full dataset
data = pd.read_csv("Static_Training_Dataset_2.csv")

# Exclude ID and Brand columns, use all other features
feature_columns = [col for col in data.columns if col not in ['ID', 'Brand']]
X = data[feature_columns]

# Scale features and train K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
data['Level'] = kmeans.fit_predict(X_scaled) + 1  # Levels 1-5

# Save the trained models and metadata
with open('warehouse_level_predictor.pkl', 'wb') as f:
    pickle.dump({
        'kmeans': kmeans,
        'scaler': scaler,
        'feature_columns': feature_columns  # Save column order
    }, f)


def predict_product_level(product_features):
    """
    Predict storage level (1-5) for a single new product.
    
    Args:
        product_features: Dict of features (excluding ID/Brand). 
        Example: {'Weight': 45, 'Height': 30, 'Fragility': 0.2}
    
    Returns:
        Predicted storage level (1-5)
    """
    # Load models
    with open('warehouse_level_predictor.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Convert input to DataFrame with correct feature order
    X_new = pd.DataFrame([product_features])[models['feature_columns']]
    
    # Scale and predict
    X_new_scaled = models['scaler'].transform(X_new)
    return models['kmeans'].predict(X_new_scaled)[0] + 1