import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
cor = '#2a77af'


def plot_pca_2d(robust_features, labels, random_state = 42):
    pca_2d = PCA(n_components = 2, random_state = random_state)
    coords = pca_2d.fit_transform(robust_features)
    
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize = (10, 6))
    
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], c = [colors[cluster_id]], label = f"Cluster {cluster_id}", alpha = 0.4, s = 10)
    
    ax.set_title("PCA 2D Projection — Customer Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title = "Cluster", markerscale = 2)
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(df, cluster_col='cluster'):
    
    cluster_pct = (df[cluster_col].value_counts(normalize=True).sort_index() * 100 )

    colors = list(plt.cm.tab10.colors)
    palette = {cluster_id: colors[cluster_id] for cluster_id in sorted(df[cluster_col].unique())}

    ax = cluster_pct.plot(kind = 'bar', figsize = (8, 5), color=[palette[c] for c in cluster_pct.index])

    plt.title('Customer Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of Customers (%)')

    for i, value in enumerate(cluster_pct):
        ax.text(i, value + 0.5, f'{value:.1f}%', ha = 'center')

    plt.tight_layout()
    plt.show()


def plot_cluster_boxplots(df, labels, features):
    colors = list(plt.cm.tab10.colors)
    palette = {cluster_id: colors[cluster_id] for cluster_id in sorted(np.unique(labels))}
    
    df_plot = df[features].copy()
    df_plot["cluster"] = labels
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        sns.boxplot(data = df_plot, x = "cluster", y = col, hue = "cluster", palette = palette, legend=False, fliersize = 0, ax = axes[i])
        axes[i].set_title(col.replace("_", " ").title(), fontsize = 10)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Feature Distribution by Cluster", fontsize = 14)
    plt.tight_layout(rect = [0, 0, 1, 0.96])
    plt.show()


def plot_cluster_discrete_means(df, labels, features):
    colors = list(plt.cm.tab10.colors)
    
    df_plot = df[features].copy()
    df_plot["cluster"] = labels
    
    cluster_means = df_plot.groupby("cluster")[features].mean()
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        cluster_ids = cluster_means.index
        values = cluster_means[col]
        bar_colors = [colors[c] for c in cluster_ids]
        axes[i].bar(cluster_ids, values, color = bar_colors, width = 0.6)
        axes[i].set_title(col, fontsize = 10)
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Mean")
        axes[i].set_xticks(cluster_ids)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Average Discrete Features by Cluster", fontsize = 14)
    plt.tight_layout()
    plt.show()


def plot_cluster_top_features(df, labels, top_n = 5):
    df_plot = df.copy()
    df_plot["cluster"] = labels
    
    numeric_cols = df_plot.select_dtypes(include = "number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster"]
    
    scaler = MinMaxScaler()
    df_normalized = df_plot.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df_plot[numeric_cols])
    
    cluster_means = df_normalized.groupby("cluster")[numeric_cols].mean()
    global_mean = df_normalized[numeric_cols].mean()
    
    cluster_diff = cluster_means - global_mean
    
    n_clusters = df_plot["cluster"].nunique()
    fig, axes = plt.subplots(1, n_clusters, figsize = (6 * n_clusters, 6))
    
    for cluster_id in sorted(df_plot["cluster"].unique()):
        ax = axes[cluster_id]
        top_features = cluster_diff.loc[cluster_id].abs().nlargest(top_n).index
        values = cluster_diff.loc[cluster_id][top_features]
        bar_colors = ["#2ea12e" if v > 0 else "#ca2915" for v in values]
        
        ax.barh([f.replace("_", " ").title() for f in top_features], values, color = bar_colors)
        ax.axvline(0, color = "black", linewidth = 0.8)
        ax.set_title(f"Cluster {cluster_id} — Top Distinguishing Features", fontsize = 11)
        ax.set_xlabel("Deviation from Global Mean")
    
    plt.suptitle("Top Distinguishing Features per Cluster", fontsize = 14)
    plt.tight_layout()
    plt.show()