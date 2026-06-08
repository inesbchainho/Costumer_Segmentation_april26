import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def print_highly_correlated_pairs(df, threshold=0.8):
    corr_matrix = df.corr(method='spearman').abs()
    high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    readable = (high_corr.stack().reset_index().rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'}))
    high_pairs = readable[readable['corr'] > threshold]
    print(high_pairs)

def plot_feature_histograms(df, columns):
    colors = sns.color_palette("husl", len(columns))
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        sns.histplot(
            df[col],
            bins=30,
            kde=True,
            ax=axes[i],
            color=colors[i]
        )
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Histograms of Key Customer Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_feature_boxplots(df, columns):
    """
    Plots boxplots of selected numeric features in a single figure.
    """
    
    colors = sns.color_palette("Set2", len(columns))
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        sns.boxplot(
            x=df[col],
            ax=axes[i],
            color=colors[i]
        )
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Boxplots of Key Customer Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
