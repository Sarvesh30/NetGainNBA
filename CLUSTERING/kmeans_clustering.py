import numpy as np
import pandas as pd
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

def perform_preprocessing(df):
    """
    Removes certain data points and irrelevant columns.
    Takes in the original dataset and returns a filtered, normalized dataset.
    """
    df = df[df['MIN'] > 5] # Remove players with insigifnicant amount of min.

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
    df = pd.DataFrame(df_scaled, columns=df.columns, index = df.index)

    return df

def display_variance_ratios(pca):
    """
    An explained variance ratio represents a proportion of a dataset's variance
    explained by each principal component.
    The higher a PC's variance ratio, the more important info it carries.
    """
    print("\nPCA Variance Ratio:")
    i = 1
    for ratio in pca.explained_variance_ratio_:
        print(f"PCA_{i}: {ratio}")
        i += 1

def display_pca_scatter_plot(pca_df):
    plt.scatter(
            pca_df[0], 
            pca_df[1], 
            color='black', 
            alpha=0.7
        )
    plt.title("PCA")
    plt.xlabel(f"PC{pca_df.columns[0] + 1}")
    plt.ylabel(f"PC{pca_df.columns[1] + 1}")
    plt.show()

def perform_PCA(df):
    """Reduces the number of dimensions of the preprocessed data set."""
    pca = PCA(n_components=2)
    pca_scaled = pca.fit_transform(df)
    print(f"Before PCA: {df.shape[1]} features")
    print(f"After PCA: {pca_scaled.shape[1]} features")

    pca_df = pd.DataFrame(pca_scaled)

    # display_variance_ratios(pca)
    display_pca_scatter_plot(pca_df)

    return pca, pca_df

def get_pca_loadings(df, pca):
    """
    Calcuates loadings to get relationship between PCs and original components.
    Returns a dataframe featuring the loadings in a stat x PC data set.
    """
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    loadings = eigenvectors * np.sqrt(eigenvalues).reshape(-1, 1)
    loadings = loadings.T # Transpose to swap rows and cols
        
    loadings_df = pd.DataFrame(loadings,
                               columns =[f'PC{i+1}' for i in range(loadings.shape[1])],
                               index=df.columns)

    return loadings_df

def determine_relevant_stats_from_loadings(loadings_df):
    """
    Determines the stat with the highest absolute loading value for each PC.
    """
    relevant_stats = loadings_df.abs().idxmax()
    relevant_stats = list(set(relevant_stats)) # To remove duplicates

    return relevant_stats

def perform_elbow_method(df):
    """Provides insight on how many clusters that should be utilized."""
    k_range = range(1, 10)
    sum_squared_errors = []
    for k in k_range:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sum_squared_errors.append(km.inertia_)

    plt.xlabel('K')
    plt.ylabel('Sum of squared errors')
    plt.title("Elbow Method")
    plt.plot(k_range, sum_squared_errors)
    plt.show()

def conduct_kmeans_clustering(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(df)
    cluster_labels = kmeans.predict(df)
    cluster_centers = kmeans.cluster_centers_

    df['Cluster'] = cluster_labels

    return cluster_labels, cluster_centers

def display_kmeans_scatter_plot(df, cluster_labels, cluster_centers, num_clusters):
    colors = cycle(cm.tab10.colors)

    for i in range(num_clusters):
        color = next(colors)
        idx = cluster_labels == i # Array has which data points belong to cluster i
        plt.scatter( # Plot cluster
            df.iloc[idx, 0], 
            df.iloc[idx, 1], 
            color=color, 
            label=f'Cluster {i+1}', 
            alpha=0.7
        )
        plt.scatter( # Plot cluster center
            cluster_centers[i, 0], 
            cluster_centers[i, 1],
            edgecolors='black',
            linewidths=2,
            color=color, 
            s=150
        )

    plt.title("K-Means Clustering")
    plt.xlabel(f"{df.columns[0]}")
    plt.ylabel(f"{df.columns[1]}")
    plt.legend()
    plt.show()

def find_silhouette_scores(df, cluster_labels, num_clusters):
    """
    Evaluates how well k-means clustering results are.
    Scores above 0.25 are good.
    """
    silhouette_vals = silhouette_samples(df, cluster_labels)
    avg_silhouette_score_per_cluster = {}

    for cluster in range(num_clusters):
        avg_silhouette_score_per_cluster[cluster] = np.mean(
            silhouette_vals[cluster_labels == cluster]
        )

    avg_silhouette_score_df = pd.DataFrame(
        avg_silhouette_score_per_cluster.items(), 
            columns=['Cluster', 'Avg. Silhouette Score']
        )
    print("\nCluster Avg. Silhouette Scores:")
    print(avg_silhouette_score_df)

def main():
    num_clusters = 5
    
    df = pd.read_csv("../NBA_PLAYER_DATA/NBA_PLAYER_FINAL_DATASET.csv")
    df = perform_preprocessing(df)

    pca, pca_df = perform_PCA(df)
    loadings_df = get_pca_loadings(df, pca)
    relevant_stats = determine_relevant_stats_from_loadings(loadings_df)
    df = df[relevant_stats]

    # perform_elbow_method(df)
    cluster_labels, cluster_centers = conduct_kmeans_clustering(df, num_clusters)
    display_kmeans_scatter_plot(df, cluster_labels, cluster_centers, num_clusters)
    # find_silhouette_scores(df, cluster_labels, num_clusters)

if __name__ == "__main__":
    main()