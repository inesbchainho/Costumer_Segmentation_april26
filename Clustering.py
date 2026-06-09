import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

cor = '#3ecae3'

def find_optimal_k(df, k_min, k_max):
    k_range = range(k_min, k_max)
    silhouette_scores = [] 
    inertia = []
    for k in k_range:
        model = KMeans(n_clusters = k, random_state = 42)
        model.fit(df)
        sil_score = silhouette_score(df, model.labels_)
        silhouette_scores.append(sil_score)
        inertia.append(model.inertia_)
        # print(f"k={k}, silhouette={sil_score:.4f}, inertia={model.inertia_:.0f}")
    return list(k_range), silhouette_scores, inertia

def plot_kmeans_scores(k_range, silhouette_scores, inertia_scores):
    plt.figure(figsize = (12, 4))
    # Silhouette - Local maximums
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker = 'o', color = cor)
    plt.title("Silhouette Score per k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    # Inertia - Elbow method
    plt.subplot(1, 2, 2)
    plt.plot(k_range, inertia_scores, marker = 'o', color = cor)
    plt.title("Inertia per k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

def plot_pca_k_grid(robust_features, variance_range, k_range, random_state=42):
    results = []

    for var in variance_range:
        pca = PCA(n_components=var, random_state=random_state)
        pca_features = pca.fit_transform(robust_features)
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=random_state).fit_predict(pca_features)
            score = silhouette_score(pca_features, labels)
            results.append({"variance": var, "k": k, "silhouette": score})

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="k", columns="variance", values="silhouette")

    fig, ax = plt.subplots(figsize=(12, 6))
    for var in variance_range:
        ax.plot(pivot.index, pivot[var], marker="o", label=f"var={var}")

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k and PCA Variance Retained")
    ax.legend(title="Variance Retained", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticks(k_range)
    plt.tight_layout()
    plt.show()

def kmeans_model(k, robust_features, var):
    pca = PCA(n_components = var, random_state = 42)
    pca_features = pca.fit_transform(robust_features)
    kmeans = KMeans(n_clusters = k, random_state = 42)
    labels = kmeans.fit_predict(pca_features)
    return labels

def plot_pca_2d(robust_features, labels, var=0.8, random_state=42):
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords = pca_2d.fit_transform(robust_features)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.4, s=10)
    plt.colorbar(scatter, label="Cluster")
    plt.title("PCA 2D Projection — Customer Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def plot_cluster_boxplots(df, labels, features):
    df_plot = df[features].copy()
    df_plot["cluster"] = labels
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        df_plot.boxplot(column=col, by="cluster", ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Feature Distribution by Cluster", fontsize=14)
    plt.tight_layout()
    plt.show()
    