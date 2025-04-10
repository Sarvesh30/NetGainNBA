import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
import time

def year_iterator(year_1_min, year_2_max):

    year_range = range(year_1_min,year_2_max)

    for y in year_range:
        year = str(y)

        next_year = y + 1  # Increment the year by 1
        next_year_last_two = str(next_year)[-2:]  # Extract the last two digits of the next year

        # Handle the case where the next year's last two digits are '00' (i.e., the year 2000, 2100, etc.)
        if next_year_last_two == '00':
            next_year_last_two = '00'

        yield f"{year}-{next_year_last_two}"

def data_filtering(year_1_min, year_2_max, df):
    filtered_df = df[(df['MIN'] > 5) & (df['GP'] > 15)]
    
    # Generate a list of valid seasons using year_iterator
    valid_seasons = list(year_iterator(year_1_min, year_2_max))

    # Filter the DataFrame to include only rows with valid seasons
    filtered_df = filtered_df[filtered_df['SEASON'].isin(valid_seasons)]

    old_count = len(df)
    new_count = len(filtered_df)

    print(f"Went from {old_count} players to {new_count} players.")
    print(f"{old_count - new_count} players were filtered out; playing less than 5 minutes a game.")

    return filtered_df

def features_selector(df, decision):
    available_features = df.select_dtypes(include=np.number).columns.tolist()

    excluded_features1 = ['SEASON', 'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ABBREVIATION', 
                          'TEAM_ID', 'AGE', 'W', 'L', 'W_PCT', 'GP', 'MIN']
    excluded_features2 = [
        "RESTRICTED_AREA_FGM", "RESTRICTED_AREA_FGA", "RESTRICTED_AREA_FG_PCT",
        "IN_THE_PAINT_(NON-RA)_FGM", "IN_THE_PAINT_(NON-RA)_FGA", "IN_THE_PAINT_(NON-RA)_FG_PCT",
        "MID-RANGE_FGM", "MID-RANGE_FGA", "MID-RANGE_FG_PCT",
        "CORNER_3_FGM", "CORNER_3_FGA", "CORNER_3_FG_PCT",
        "ABOVE_THE_BREAK_3_FGM", "ABOVE_THE_BREAK_3_FGA", "ABOVE_THE_BREAK_3_FG_PCT"
    ]

    if decision == 1:
        excluded_features = excluded_features1
    elif decision == 2:
        excluded_features = excluded_features1 + excluded_features2
    elif decision == 3:
        #included features are meant to be manually inputed
        included_features = ['OREB_PCT', 'DREB_PCT', 'REB_PCT',  # Rebounding percentage (defensive & overall)
        'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR',  
        'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_PAINT',  # Shot selection & scoring zones
        'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM',  
        'PCT_AST_FGM', 'PCT_UAST_FGM',  # Assisted vs. unassisted shots
        'RESTRICTED_AREA_FGA/FGA', 'IN_THE_PAINT_(NON-RA)_FGA/FGA',  
        'MID-RANGE_FGA/FGA', 'CORNER_3_FGA/FGA', 'ABOVE_THE_BREAK_3_FGA/FGA', 
        'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'AST_RATIO']

    elif decision == 4:
        included_features = ['OREB_PCT', 'DREB_PCT', 'REB_PCT',  # Rebounding percentage (defensive & overall)
        'PCT_FGA_2PT', 'PCT_FGA_3PT',  
        'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM',  
        'PCT_AST_FGM', 'PCT_UAST_FGM',  # Assisted vs. unassisted shots
        'RESTRICTED_AREA_FGA/FGA', 'IN_THE_PAINT_(NON-RA)_FGA/FGA',  
        'MID-RANGE_FGA/FGA', 'CORNER_3_FGA/FGA', 'ABOVE_THE_BREAK_3_FGA/FGA', 
        'AST_RATIO', 'STL/MIN', 'BLK/MIN']

    else:
        raise ValueError("Invalid decision.")
    
    if decision == 1 or decision == 2:
        available_features = [col for col in available_features if col not in excluded_features]
    elif decision == 3 or decision == 4:
        available_features = [col for col in available_features if col in included_features]
    else:
        raise ValueError("Invalid decision.")

    if len(available_features) < 5:
        raise ValueError("Insufficient features selected for clustering. Check dataset or feature selection logic.")

    return available_features, excluded_features1

def scale_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return pca_data, explained_variance, pca

def find_optimal_clusters(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='random', random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append((k, score))
    return silhouette_scores

def visualize_clusters(year_range, pca_data, labels, centroids=None, save_path="player_clusters.png"):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='viridis', s=50)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f"{year_range} Player Clusters Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def visualize_silhouette_scores(scores, save_path="silhouette_scores.png"):
    plt.figure(figsize=(10, 6))
    ks, sil_scores = zip(*scores)
    plt.plot(ks, sil_scores, marker='o')
    plt.title("Silhouette Scores for Different Cluster Counts")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.savefig(save_path)
    plt.show()

def display_pca_feature_importance(pca, feature_names, n_components=2, top_only=False):
    for i in range(n_components):
        component = pca.components_[i]
        feature_importance = sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nFeatures for PCA Component {i + 1}:")
        for feature, loading in feature_importance:
            print(f"{feature}: {loading:.4f}")
        if top_only:
            print("\nTop features only:")
            for feature, loading in feature_importance[:5]:  # Display top 5 features
                print(f"{feature}: {loading:.4f}")

def filter_features_by_pca_importance(pca, feature_names, n_components=2, threshold=0.1):

    
    # Calculate the absolute loading scores for the first n_components
    importance_scores = np.abs(pca.components_[:n_components]).mean(axis=0)

    # df of feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })

    # Filter features based on mean threshold
    filtered_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

    print("\nFiltered Features Based on PCA Importance:")
    print(filtered_features)

    return filtered_features

def single_cluster_generator(df, n_clusters):
    """
    Generates and saves individual cluster-specific CSV files.
    """
    df = df.copy()
    n_clusters = int(n_clusters)

    # Ensure the output directory exists
    output_dir = 'NetGainNBA/Player Clusters/'
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrames for each cluster
    df_list = [df[df['CLUSTER'] == i] for i in range(n_clusters)]

    for i, cluster_df in enumerate(df_list):
        if cluster_df.empty:
            print(f"No players in cluster {i}.")
        else:
            # Save the cluster-specific DataFrame to a CSV file
            cluster_filename = os.path.join(output_dir, f'NBA_PLAYER_CLUSTER_{i}.csv')
            cluster_df.to_csv(cluster_filename, index=False)
            print(f"Saving cluster {i} to {cluster_filename}.")
            
    return df_list

def cluster_means_calculator(df_list, included_features):
    
    df_list = df_list
    included_features = included_features

    # Iterate through the df_list and calculate mean values for included features
    cluster_means = {}
    
    for i, df in enumerate(df_list):
        # Ensure the DataFrame contains the included features
        available_features = [feature for feature in included_features if feature in df.columns]
        
        # Calculate the mean for the available features
        cluster_means[f"Cluster_{i}"] = df[available_features].mean()

    # Convert the cluster_means dictionary to a DataFrame for better visualization
    cluster_means_df = pd.DataFrame(cluster_means)

    # Transpose the DataFrame to make clusters the rows
    cluster_means_df = cluster_means_df.T

    print(cluster_means_df)
    return cluster_means_df

def cluster_z_calculator(df):
    """
    Calculates z-scores for numeric columns in the DataFrame and fixes the CLUSTER column.
    """
    # Ensure numeric columns are selected
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate z-scores
    mean = np.mean(numeric_df, axis=0)
    st_dev = np.std(numeric_df, axis=0)
    z_scores = (numeric_df - mean) / st_dev
    z_scores = z_scores.round(4)

    # Fix the CLUSTER column by stripping "Cluster_" and converting to integers
    df["CLUSTER"] = df["CLUSTER"].str.replace("Cluster_", "").astype(int)

    # Combine the fixed CLUSTER column with the z-scores
    z_scores_df = pd.concat([df["CLUSTER"], z_scores], axis=1)

    # Save the z-scores DataFrame to a CSV file
    z_scores_df.to_csv("NetGainNBA/Player Clusters/Cluster_Z_Scores.csv", index=False)

    print("\nZ-Scores:")
    print(z_scores_df)
    return z_scores_df

def cluster_name_calculator(df):
    """
    Assigns a name to each cluster based on its characteristics, prints the name, 
    and returns a mapping of cluster IDs to names.
    """
    cluster_name_mapping = {}

    for cluster in df["CLUSTER"].unique():
        # Get the row corresponding to the current cluster
        cluster_row = df[df["CLUSTER"] == cluster]
        
        if not cluster_row.empty:
            # Determine the cluster name based on its characteristics
            if (cluster_row['PCT_FGA_3PT'].values[0] > cluster_row['PCT_FGA_2PT'].values[0] and
                cluster_row['PCT_FGA_3PT'].values[0] > 1.50 and
                cluster_row['PCT_AST_FGM'].values[0] > 1.00 and
                cluster_row['REB_PCT'].values[0] < -.50):
                cluster_name = "3&D SPECIALIST"
            elif (0 < cluster_row['PCT_FGA_3PT'].values[0] < .571 and
                  -.571 < cluster_row['PCT_FGA_2PT'].values[0] < 0 and
                  .50 < cluster_row['PCT_UAST_FGM'].values[0] < 1.05 and
                  cluster_row['AST_RATIO'].values[0] > 1.25 and
                  1 < cluster_row['PCT_UAST_2PM'].values[0] < 1.41):
                cluster_name = "VERSATILE PLAYMAKER"
            elif (-.05 < cluster_row['MID-RANGE_FGA/FGA'].values[0] and
                  -.5 < cluster_row['PCT_FGA_2PT'].values[0] < .57 and
                  -.57 < cluster_row['PCT_FGA_3PT'].values[0] < .50 and
                  -.01 < cluster_row['PCT_UAST_2PM'].values[0] < .50 and
                  cluster_row['REB_PCT'].values[0] < 0):
                cluster_name = "BALANCED SHOT CREATOR"
            elif (cluster_row['PCT_FGA_3PT'].values[0] < -1.0 and
                  cluster_row['PCT_FGA_2PT'].values[0] > 1.00 and
                  .71 < cluster_row['REB_PCT'].values[0] < 1.4 and
                  (-.28 < cluster_row['PCT_UAST_2PM'].values[0] < .45 or -1.3 < cluster_row['PCT_UAST_2PM'].values[0] < .25)):
                cluster_name = "INTERIOR SCORER"
            elif (cluster_row['PCT_FGA_3PT'].values[0] < -1.0 and
                  cluster_row['PCT_FGA_2PT'].values[0] > 1.0 and
                  cluster_row['REB_PCT'].values[0] > 1.4 and
                  cluster_row['RESTRICTED_AREA_FGA/FGA'].values[0] > 1.75):
                cluster_name = "RIM-RUNNER"
            elif (cluster_row['CORNER_3_FGA/FGA'].values[0] > 0 and
                  cluster_row['PCT_FGA_3PT'].values[0] > -.1 and
                  -.50 < cluster_row['REB_PCT'].values[0] < .50 and
                  cluster_row['PCT_AST_FGM'].values[0] > 0.6 and
                  cluster_row['RESTRICTED_AREA_FGA/FGA'].values[0] < 0):
                cluster_name = "FLOOR STRETCHER"
            elif (.25 < cluster_row['PCT_FGA_2PT'].values[0] < 1 and
                  -1 < cluster_row['PCT_FGA_3PT'].values[0] < -.25 and
                  cluster_row['PCT_AST_FGM'].values[0] > -.06 and
                  cluster_row['REB_PCT'].values[0] > 0):
                cluster_name = "HYBRID BIG"
            elif (cluster_row['PCT_UAST_FGM'].values[0] > 1.5 and
                  cluster_row['AST_RATIO'].values[0] > 1.5 and
                  cluster_row['REB_PCT'].values[0] < 0):
                cluster_name = "ISOLATION SCORER"
            elif (-1.5 < cluster_row['PCT_FGA_2PT'].values[0] < -.75 and
                  .75 < cluster_row['PCT_FGA_3PT'].values[0] < 1.5 and
                  cluster_row['PCT_AST_FGM'].values[0] > cluster_row['PCT_UAST_FGM'].values[0] and
                  cluster_row['IN_THE_PAINT_(NON-RA)_FGA/FGA'].values[0] < -.50):
                cluster_name = "OFF-BALL SHOOTER"
            else:
                cluster_name = "UNKNOWN"

            # Print the cluster name
            print(f"Cluster {cluster} is a {cluster_name}.")
            cluster_name_mapping[cluster] = cluster_name
        else:
            print(f"No data found for Cluster {cluster}.")
            cluster_name_mapping[cluster] = "UNKNOWN"

    return cluster_name_mapping

def main():
    # Load the dataset
    read_csv = pd.read_csv('NetGainNBA/Project Codes/NBA_PLAYER_FINAL_DATASET.csv')
    df = read_csv.copy()

    # Define the year range for filtering
    year_1_min = 1996
    year_2_max = 2024

    # Filter the data
    filtered_df = data_filtering(year_1_min, year_2_max, df)

    # Manual decision for testing
    decision = 4
    selected_features, key_features = features_selector(filtered_df, decision=decision)
    print(f"Selected features: {selected_features}")

    # Prepare data
    df_selected = filtered_df[selected_features]
    scaled_data, scaler = scale_data(df_selected)

    # PCA
    pca_data, explained_variance, pca = perform_pca(scaled_data)
    print(f"Explained variance by PCA components: {explained_variance}")

    # Visualize PCA feature importance
    display_pca_feature_importance(pca, selected_features, n_components=2)

    # Clustering
    n_clusters = 9
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)

    # Add cluster labels to the original DataFrame
    filtered_df['CLUSTER'] = labels
    print(filtered_df['CLUSTER'].value_counts())

    filtered_df = filtered_df[key_features + selected_features + ['CLUSTER']]

    # Visualize clusters
    year_range = f"{year_1_min}-{year_2_max}"
    visualize_clusters(year_range, pca_data, labels, centroids=kmeans.cluster_centers_)

    # Specific cluster CSVs
    df_list = single_cluster_generator(filtered_df, n_clusters)

    # Use the list of DataFrames for further processing
    cluster_means_df = cluster_means_calculator(df_list, selected_features)
    cluster_means_df.index.name = "CLUSTER"
    cluster_means_df = cluster_means_df.reset_index()

    # Calculate z-scores
    z_scores_df = cluster_z_calculator(cluster_means_df)

    # Add cluster names to the DataFrame
    cluster_name_mapping = cluster_name_calculator(z_scores_df)
    filtered_df['CLUSTER_NAME'] = filtered_df['CLUSTER'].map(cluster_name_mapping)

    # Save the updated DataFrame with cluster names
    filtered_df.to_csv('NetGainNBA/Project Codes/NBA_PLAYER_CLUSTERED.csv', index=False)
    
    

if __name__ == "__main__":
    main()

