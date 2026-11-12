"""
baselines.py

What is done here:
    - Baseline models are used to compare our main model (BiLSTM)
        - K-Means: Model will group customers based on variables
        - Isolation Forest: Tree-based model designed to find anomalies
THIS IS USED FOR THE REPORT
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataset import CSVDataset

def baselines():
    x_tensor, customer_ids, _ = CSVDataset()

    # ISOLATION FOREST:
    # Turn our 3D BILSTM [Batch, 50, 6] INTO 2D for Sklearn (50 * 6) -> [Batch, 300]
    x = x_tensor.numpy().reshape(len(customer_ids), -1)
    # Scale
    x_scaled = StandardScaler().fit_transform(x)

    # Initialize Model
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(x_scaled)
    # Lower score = More Anomalous, so here we flip
    results_iso = -iso.decision_function(x_scaled)

    # K-MEANS CLUSTERING
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(x_scaled)
    centers = kmeans.cluster_centers_
    results_kmeans = np.linalg.norm(x_scaled-centers[clusters], axis=1)

    overall_results = pd.DataFrame({
        "Customer ID": customer_ids,
        "Isolation Forest Score": results_iso,
        "K-Means Score": results_kmeans,
        "K-Means Cluster": clusters
    })

    return overall_results
