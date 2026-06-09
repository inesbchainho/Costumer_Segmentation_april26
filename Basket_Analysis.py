import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def plot_top_products(exploded_basket, n_top=10):
    colors = list(plt.cm.tab10.colors)
    cluster_ids = sorted(exploded_basket["cluster"].unique())
    n_clusters = len(cluster_ids)
    
    n_cols = 2
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    axes = axes.flatten()
    
    for i, cluster_id in enumerate(cluster_ids):
        ax = axes[i]
        top_products = (exploded_basket[exploded_basket["cluster"] == cluster_id]["product"]
                       .value_counts()
                       .head(n_top))
        top_products.plot(kind="barh", ax=ax, color=colors[cluster_id])
        ax.invert_yaxis()
        ax.set_title(f"Top Products for Cluster {cluster_id}", fontsize=12)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Product")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Top Products per Cluster", fontsize=16)
    plt.tight_layout()
    plt.show()


def basket_analysis(exploded_basket, min_support = 0.01, min_threshold = 1.0):
    rules_by_cluster = {}
    
    for cluster_id in sorted(exploded_basket["cluster"].unique()):
        cluster_baskets = (exploded_basket[exploded_basket["cluster"] == cluster_id].groupby("invoice_id")["product"].apply(list).tolist())
        
        te = TransactionEncoder()
        te_array = te.fit_transform(cluster_baskets)
        basket_df = pd.DataFrame(te_array, columns = te.columns_)
        
        frequent_items = apriori(basket_df, min_support = min_support, use_colnames = True)
        rules = association_rules(frequent_items, metric = "lift", min_threshold = min_threshold)
        rules = rules.sort_values("lift", ascending = False)
        
        rules_by_cluster[cluster_id] = rules
        # print(f"=== Cluster {cluster_id} === {len(rules)} rules found")
        # print(rules.head(top_n)[["antecedents", "consequents", "support", "confidence", "lift"]])
        # print()
    
    return rules_by_cluster


def plot_association_rules(rules_by_cluster, top_n = 10):
    colors = list(plt.cm.tab10.colors)
    cluster_ids = sorted(rules_by_cluster.keys())
    n_clusters = len(cluster_ids)
    
    n_cols = 2
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (20, 8 * n_rows))
    axes = axes.flatten()
    
    for i, cluster_id in enumerate(cluster_ids):
        ax = axes[i]
        rules = rules_by_cluster[cluster_id]
        top_rules = rules.nlargest(top_n, "lift")
        
        labels = [f"{', '.join(list(a))} → {', '.join(list(c))}" for a, c in zip(top_rules["antecedents"], top_rules["consequents"])]
        
        ax.barh(range(len(labels)), top_rules["lift"].values, color = colors[cluster_id])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize = 10)
        ax.invert_yaxis()
        ax.set_title(f"Top Rules by Lift — Cluster {cluster_id}", fontsize = 13)
        ax.set_xlabel("Lift")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Top Association Rules per Cluster", fontsize = 16)
    plt.tight_layout(rect = [0, 0, 1, 0.96])
    plt.show()