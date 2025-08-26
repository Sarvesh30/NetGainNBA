import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

def team_proportion_calculator(player_df, team_df):
    # Select relevant columns
    player_key_columns = ['SEASON', 'TEAM_ID', 'GP', 'MIN', 'CLUSTER_NAME']
    player_df = player_df[player_key_columns]
    
    cluster_columns = ["3&D SPECIALIST", "VERSATILE PLAYMAKER", "BALANCED SHOT CREATOR",
                       "INTERIOR SCORER", "RIM-RUNNER", "FLOOR STRETCHER", "HYBRID BIG",
                       "ISOLATION SCORER", "OFF-BALL SHOOTER"]

    v_e_metrics = [
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "AST", "OREB", "DREB", "REB",
    "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PLUS_MINUS", "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE", "OPP_PTS_FB",
    "OPP_PTS_PAINT", "E_OFF_RATING", "OFF_RATING", "E_DEF_RATING", "DEF_RATING", "E_NET_RATING", "NET_RATING",
    "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "E_PACE",
    "PACE", "PACE_PER40", "POSS", "PIE", "PTS_OFF_TOV", "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT", "PCT_FGA_2PT",
    "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_2PT_MR", "PCT_PTS_3PT", "PCT_PTS_FB", "PCT_PTS_FT", "PCT_PTS_OFF_TOV",
    "PCT_PTS_PAINT", "PCT_AST_2PM", "PCT_UAST_2PM", "PCT_AST_3PM", "PCT_UAST_3PM", "PCT_AST_FGM", "PCT_UAST_FGM",
    "RESTRICTED_AREA_FGM", "RESTRICTED_AREA_FGA", "RESTRICTED_AREA_FG_PCT", "IN_THE_PAINT_(NON-RA)_FGM",
    "IN_THE_PAINT_(NON-RA)_FGA", "IN_THE_PAINT_(NON-RA)_FG_PCT", "MID-RANGE_FGM", "MID-RANGE_FGA", "MID-RANGE_FG_PCT",
    "CORNER_3_FGM", "CORNER_3_FGA", "CORNER_3_FG_PCT", "ABOVE_THE_BREAK_3_FGM", "ABOVE_THE_BREAK_3_FGA",
    "ABOVE_THE_BREAK_3_FG_PCT"] # volume and efficiency metrics

    team_columns = ['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GP', 'MIN'] + v_e_metrics  # Include v_e_metrics columns
    team_df = team_df[team_columns]

    # Calculate weighted minutes for each player (MIN * GP)
    player_df['WEIGHTED_MIN'] = player_df['GP'] * player_df['MIN']

    # Group by SEASON, TEAM_ID, and CLUSTER_NAME to sum weighted minutes for each cluster
    cluster_weighted_minutes = (
        player_df.groupby(['SEASON', 'TEAM_ID', 'CLUSTER_NAME'])['WEIGHTED_MIN']
        .sum()
        .reset_index()
        
    )

    print('Cluster Minutes:')
    print(cluster_weighted_minutes.head())
    cluster_weighted_minutes.to_csv('NetGainNBA/Player Clusters/CLUSTER_MINUTES.csv', index=False)

    # Calculate total team minutes by summing all players' weighted minutes for each team
    total_team_minutes = (
        player_df.groupby(['SEASON', 'TEAM_ID'])['WEIGHTED_MIN']
        .sum()
        .reset_index()
        .rename(columns={'WEIGHTED_MIN': 'TOTAL_TEAM_MIN'})
    )

    print('Total Team Minutes:')
    print(total_team_minutes.head())  # Debugging line to check the total team minutes
    total_team_minutes.to_csv('NetGainNBA/Team Clusters/TOTAL_TEAM_MINUTES.csv', index=False)

    # Merge total team minutes with cluster weighted minutes
    cluster_weighted_minutes = cluster_weighted_minutes.merge(
        total_team_minutes,
        on=['SEASON', 'TEAM_ID'],
        how='left'
    )

    # Calculate the proportion of minutes played by each cluster
    cluster_weighted_minutes['PROPORTION'] = (
        cluster_weighted_minutes['WEIGHTED_MIN'] / cluster_weighted_minutes['TOTAL_TEAM_MIN']
    ).round(4)

    # Pivot the table to make each cluster name a column
    pivoted_cluster_proportions = cluster_weighted_minutes.pivot_table(
        index=['SEASON', 'TEAM_ID'],  # Rows: TEAM_ID and SEASON
        columns='CLUSTER_NAME',      
        values='PROPORTION',         
        fill_value=0
    ).reset_index()                  # Reset index to make TEAM_ID and SEASON columns

    # Merge the pivoted proportions with the original team_df to include GP, MIN, and v_e_metrics
    final_proportions = pivoted_cluster_proportions.merge(
        team_df,
        on=['SEASON', 'TEAM_ID'],
        how='left'
    )

    proportions = final_proportions[['SEASON', 'TEAM_ID', 'GP', 'MIN'] + cluster_columns]

    # Save the result to a CSV file
    proportions.to_csv('NetGainNBA/Team Clusters/TEAM_PLAYSTYLE_PROPORTIONS.csv', index=False)

    # Print the result
    print("Team Playstyle Proportions:")
    print(proportions.head())

    cluster_columns = ["3&D SPECIALIST", "VERSATILE PLAYMAKER", "BALANCED SHOT CREATOR",
                       "INTERIOR SCORER", "RIM-RUNNER", "FLOOR STRETCHER", "HYBRID BIG",
                       "ISOLATION SCORER", "OFF-BALL SHOOTER"]

    # Use final_proportions as clustering_metrics
    clustering_metrics = final_proportions

    return clustering_metrics, cluster_columns, team_columns, v_e_metrics

def player_cluster_counter(player_df, team_df):
    player_key_columns = ['SEASON', 'TEAM_ID', 'CLUSTER_NAME']
    player_df = player_df[player_key_columns]
    


    team_columns = ['SEASON', 'TEAM_NAME', 'TEAM_ID']
    team_df = team_df[team_columns]

    cluster_columns = ["3&D SPECIALIST", "VERSATILE PLAYMAKER", "BALANCED SHOT CREATOR",
                       "INTERIOR SCORER", "RIM-RUNNER", "FLOOR STRETCHER", "HYBRID BIG",
                       "ISOLATION SCORER", "OFF-BALL SHOOTER"]
    
    # Group by SEASON, TEAM_ID, and CLUSTER_NAME to count occurrences
    cluster_counts = (
        player_df.groupby(['SEASON', 'TEAM_ID', 'CLUSTER_NAME'])
        .size()
        .reset_index(name='COUNT')  # Reset index and name the count column
    )

    # Pivot the table to make each cluster name a column
    pivoted_cluster_counts = cluster_counts.pivot_table(
        index=['SEASON', 'TEAM_ID'],  # Rows: TEAM_ID and SEASON
        columns='CLUSTER_NAME',      # Columns: CLUSTER_NAME
        values='COUNT',              # Values: COUNT of each cluster
        fill_value=0                 # Fill missing values with 0
    ).reset_index()                  # Reset index to make TEAM_ID and SEASON columns

    # Calculate the total count of clusters for each team and season
    pivoted_cluster_counts['TOTAL_COUNT'] = pivoted_cluster_counts[cluster_columns].sum(axis=1)

    # Convert counts to percentages (proportions)
    for cluster in cluster_columns:
        pivoted_cluster_counts[cluster] = (
            pivoted_cluster_counts[cluster] / pivoted_cluster_counts['TOTAL_COUNT']
        ).round(3)

    # Drop the TOTAL_COUNT column if no longer needed
    pivoted_cluster_counts.drop(columns=['TOTAL_COUNT'], inplace=True)

    # Save the result to a CSV file
    pivoted_cluster_counts.to_csv('NetGainNBA/Team Clusters/TEAM_PLAYER_CLUSTER_PROPORTIONS.csv', index=False)

    # Print the result
    print(pivoted_cluster_counts.head())

    return pivoted_cluster_counts, cluster_columns, team_columns 

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

def visualize_clusters(year_range, pca_data, labels, centroids=None, save_path="team_clusters.png"):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='viridis', s=50)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f"{year_range} Teams Clusters Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def visualize_silhouette_scores(scores, save_path="team_silhouette_scores.png"):
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

def main():

    player_df = pd.read_csv('NetGainNBA/Project Codes/NBA_PLAYER_CLUSTERED.csv')
    team_df = pd.read_csv('NetGainNBA/Project Codes/NBA_TEAM_FINAL_DATASET.csv')

    counted_clusters_df, selected_features, key_features = player_cluster_counter(
                            pd.read_csv('NetGainNBA/Project Codes/NBA_PLAYER_CLUSTERED.csv'), 
                           pd.read_csv('NetGainNBA/Project Codes/NBA_TEAM_FINAL_DATASET.csv'))

    print(f'The selected features are: {selected_features}')
    print(f'The key features are: {key_features}')
    print(counted_clusters_df.head())

    team_proportions_df, selected_features, key_features, v_e_features = team_proportion_calculator(
        player_df, team_df)

    # Scale data with available columns
    scaled_data, scaler = scale_data(team_proportions_df[selected_features]) #v_e_features

    pca_data, explained_variance, pca = perform_pca(scaled_data)
    print(f"Explained variance by PCA components: {explained_variance}")

    display_pca_feature_importance(pca, selected_features, n_components=2) #v_e_features

    # Clustering
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)
    
    team_proportions_df['CLUSTER'] = labels
    print(team_proportions_df['CLUSTER'].value_counts())

    team_proportions_df = team_proportions_df[key_features + selected_features + v_e_features + ['CLUSTER']]

    visualize_silhouette_scores(find_optimal_clusters(scaled_data, max_k=15))

    # Visualize clusters
    # Extract the numeric portion of the SEASON column (e.g., '1996' from '1996-97')
    team_proportions_df['SEASON_START'] = team_proportions_df['SEASON'].str[:4].astype(int)
    
    # Get the minimum and maximum years
    year_1_min = team_proportions_df['SEASON_START'].min()
    year_2_max = team_proportions_df['SEASON_START'].max() + 1  # Add 1 to include the last year in the range
    year_range = f"{year_1_min}-{year_2_max}"
    visualize_clusters(year_range, pca_data, labels, centroids=kmeans.cluster_centers_)
    team_proportions_df.drop(columns=['SEASON_START'], inplace=True)
    team_proportions_df = team_proportions_df.loc[:, ~team_proportions_df.columns.duplicated()]
    
    print("Teams Clustered:")
    print(team_proportions_df.head())
    
    # Save the clustered data to a CSV file
    team_proportions_df.to_csv('NetGainNBA/Team Clusters/TEAMS_CLUSTERED.csv', index=False)

    proportion_means_columns = ['3&D SPECIALIST','VERSATILE PLAYMAKER','BALANCED SHOT CREATOR','INTERIOR SCORER',
    'RIM-RUNNER','FLOOR STRETCHER','HYBRID BIG','ISOLATION SCORER','OFF-BALL SHOOTER','CLUSTER']

    # df of each clusters average playstyle proportions
    proportion_means = team_proportions_df[proportion_means_columns].groupby('CLUSTER').mean().round(4)

    proportion_means.to_csv('NetGainNBA/Team Clusters/CLUSTER_PROPORTION_MEANS.csv', index=True)

    # Exclude non-numeric or unwanted columns
    exclude_columns = ['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GP', 'CLUSTER'] + selected_features
    numeric_columns = team_proportions_df.columns.difference(exclude_columns)

    # Group by 'CLUSTER' and calculate the mean for numeric columns
    cluster_means = team_proportions_df.groupby('CLUSTER')[numeric_columns].mean().round(4)
    
    # Save the result to a CSV file
    cluster_means.to_csv('NetGainNBA/Team Clusters/TEAM_CLUSTER_MEANS.csv', index=True)

    # Print the result
    print("Cluster Stats Means:")
    print(cluster_means.head())

    


if __name__ == "__main__":
    main()