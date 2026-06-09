import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

cor = '#3ecae3'

def plot_pca_2d(robust_features, labels, random_state = 42):
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords = pca_2d.fit_transform(robust_features)
    
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   c=[colors[cluster_id]], label=f"Cluster {cluster_id}", 
                   alpha=0.4, s=10)
    
    ax.set_title("PCA 2D Projection — Customer Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", markerscale=2)
    plt.tight_layout()
    plt.show()

def plot_cluster_boxplots(df, labels, features):
    colors = list(plt.cm.tab10.colors)
    palette = {cluster_id: colors[cluster_id] for cluster_id in sorted(np.unique(labels))}
    
    df_plot = df[features].copy()
    df_plot["cluster"] = labels
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        sns.boxplot(data=df_plot, x="cluster", y=col, hue="cluster", palette=palette, legend=False, ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Feature Distribution by Cluster", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_cluster_discrete_means(df, labels, features):
    colors = list(plt.cm.tab10.colors)
    
    df_plot = df[features].copy()
    df_plot["cluster"] = labels
    
    cluster_means = df_plot.groupby("cluster")[features].mean()
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        cluster_ids = cluster_means.index
        values = cluster_means[col]
        bar_colors = [colors[c] for c in cluster_ids]
        axes[i].bar(cluster_ids, values, color=bar_colors, width=0.6)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Mean")
        axes[i].set_xticks(cluster_ids)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Average Discrete Features by Cluster", fontsize=14)
    plt.tight_layout()
    plt.show()