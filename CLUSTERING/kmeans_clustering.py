import numpy as np
import pandas as pd
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

def conduct_preprocessing(df):
    # Remove players with insigifnicant amount of minutes (5 or less) from the df
    df = df[df['MIN'] > 20]

    # Remove non-numerical/irrelevant columns for clustering
    df = df.drop(columns=[
        'SEASON', 
        'PLAYER_ID', 
        'PLAYER_NAME', 
        'TEAM_ID', 
        'TEAM_ABBREVIATION', 
        'AGE', 
        'GP', 
        'W', 
        'L', 
        'W_PCT', 
        'MIN']
    )

    # Using StandardScaler to normalize the varying types of stat
    # and their widely varying amounts for proper scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled

def conduct_PCA():
    pca = PCA(n_components=2)
    pca_scaled = pca.fit_transform(df_scaled)
    print(f"Before PCA: {df_scaled.shape[0]} rows, {df_scaled.shape[1]} cols")
    print(f"After PCA: {pca_scaled.shape[0]} rows, {pca_scaled.shape[1]} cols")

    df_pca = pd.DataFrame(pca_scaled)

    return df_pca

def use_elbow_method():
    k_range = range(1, 10)
    sum_squared_errors = []
    for k in k_range:
        km = KMeans(n_clusters=k)
        km.fit(df_pca)
        sum_squared_errors.append(km.inertia_)

    # Display elbow method plot
    plt.xlabel('K')
    plt.ylabel('Sum of squared errors')
    plt.title("Elbow Method")
    plt.plot(k_range, sum_squared_errors)
    plt.show()

num_clusters = 5 # input number of clusters to use after elbow method

def conduct_kmeans_clustering(df_pca):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(df_pca)
    cluster_labels = kmeans.predict(df_pca)
    cluster_centers = kmeans.cluster_centers_

    df_pca['Cluster'] = cluster_labels

    return cluster_labels, cluster_centers

def display_kmeans_scatter_plot(cluster_labels, cluster_centers):
    colors = cycle(cm.tab10.colors)

    for i in range(num_clusters):
        color = next(colors)
        idx = cluster_labels == i # array has which data points belong to cluster i
        plt.scatter( # plot cluster
            df_pca.iloc[idx, 0], 
            df_pca.iloc[idx, 1], 
            color=color, 
            label=f'Cluster {i+1}', 
            alpha=0.7
        )
        plt.scatter( # plot cluster center
            cluster_centers[i, 0], 
            cluster_centers[i, 1],
            edgecolors='black',
            linewidths=2,
            color=color, 
            s=150
        )

    plt.title("K-Means Clustering After PCA")
    plt.xlabel(f"PC{df_pca.columns[0] + 1}")
    plt.ylabel(f"PC{df_pca.columns[1] + 1}")
    plt.legend()
    plt.show()

def find_silhouette_scores(cluster_labels):
    silhouette_vals = silhouette_samples(df_pca, cluster_labels)
    avg_silhouette_score_per_cluster = {}

    for cluster in range(num_clusters):
        avg_silhouette_score_per_cluster[cluster] = np.mean(
            silhouette_vals[cluster_labels == cluster]
        )

    avg_silhouette_score_df = pd.DataFrame(
        avg_silhouette_score_per_cluster.items(), 
            columns=['Cluster', 'Avg. Silhouette Score']
        )
    print(avg_silhouette_score_df)


df = pd.read_csv("../NBA_PLAYER_DATA/NBA_PLAYER_FINAL_DATASET.csv")

df_scaled = conduct_preprocessing(df)
df_pca = conduct_PCA()
use_elbow_method()
cluster_labels, cluster_centers = conduct_kmeans_clustering(df_pca)
display_kmeans_scatter_plot(cluster_labels, cluster_centers)