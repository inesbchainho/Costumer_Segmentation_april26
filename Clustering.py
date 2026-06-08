import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
