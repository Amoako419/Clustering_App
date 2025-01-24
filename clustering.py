import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, AffinityPropagation, Birch
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

# Title and description
st.title("Clustering Algorithms Playground")
st.write("""
Explore different clustering algorithms on a synthetic dataset.
Adjust parameters interactively and visualize the results in 2D or 3D!
""")

# Generate synthetic data
centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 1]]  # 3D centers
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0, n_features=3)
X = StandardScaler().fit_transform(X)

# Display the dataset
st.subheader("Generated Dataset")
st.write("This dataset has 3 features for 3D visualization.")

# 2D Visualization
st.subheader("2D Scatter Plot")
fig_2d, ax_2d = plt.subplots()
ax_2d.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolor='black', s=50)
ax_2d.set_title("2D View of the Data (Feature 1 vs Feature 2)")
ax_2d.set_xlabel("Feature 1")
ax_2d.set_ylabel("Feature 2")
st.pyplot(fig_2d)

# 3D Visualization
st.subheader("3D Scatter Plot")
fig_3d = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels_true, title="3D View of the Data")
st.plotly_chart(fig_3d)

# Sidebar for algorithm selection
st.sidebar.title("Clustering Algorithm")
algorithm = st.sidebar.selectbox(
    "Choose an algorithm:",
    ["KMeans", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models (GMM)",
     "Spectral Clustering", "OPTICS", "Mean Shift", "Affinity Propagation", "BIRCH"]
)

# Clustering Logic
if algorithm == "KMeans":
    st.subheader("KMeans Clustering")
    n_clusters = st.slider("Number of clusters (k):", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

elif algorithm == "DBSCAN":
    st.subheader("DBSCAN Clustering")
    eps = st.slider("Epsilon (eps):", 0.1, 1.0, 0.3, step=0.1)
    min_samples = st.slider("Minimum samples:", 5, 50, 10)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

elif algorithm == "Hierarchical Clustering":
    st.subheader("Hierarchical Clustering")
    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
    linkage_method = st.selectbox("Linkage method:", ["ward", "complete", "average", "single"])
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = hierarchical.fit_predict(X)

elif algorithm == "Gaussian Mixture Models (GMM)":
    st.subheader("Gaussian Mixture Models (GMM)")
    n_components = st.slider("Number of components (clusters):", 2, 10, 3)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    labels = gmm.fit_predict(X)

elif algorithm == "Spectral Clustering":
    st.subheader("Spectral Clustering")
    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = spectral.fit_predict(X)

elif algorithm == "OPTICS":
    st.subheader("OPTICS Clustering")
    min_samples = st.slider("Minimum samples:", 5, 50, 10)
    optics = OPTICS(min_samples=min_samples)
    labels = optics.fit_predict(X)

elif algorithm == "Mean Shift":
    st.subheader("Mean Shift Clustering")
    bandwidth = st.slider("Bandwidth:", 0.1, 1.0, 0.3, step=0.1)
    mean_shift = MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(X)

elif algorithm == "Affinity Propagation":
    st.subheader("Affinity Propagation")
    damping = st.slider("Damping factor:", 0.5, 0.99, 0.9, step=0.01)
    affinity = AffinityPropagation(damping=damping, random_state=0)
    labels = affinity.fit_predict(X)

elif algorithm == "BIRCH":
    st.subheader("BIRCH Clustering")
    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
    birch = Birch(n_clusters=n_clusters)
    labels = birch.fit_predict(X)

# Plot Results
st.subheader("Clustering Results")

# 2D Plot
fig_2d, ax_2d = plt.subplots()
ax_2d.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='black', s=50)
ax_2d.set_title(f"{algorithm} Clustering (2D)")
ax_2d.set_xlabel("Feature 1")
ax_2d.set_ylabel("Feature 2")
st.pyplot(fig_2d)

# 3D Plot
fig_3d = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels, title=f"{algorithm} Clustering (3D)")
st.plotly_chart(fig_3d)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ by Amoako")