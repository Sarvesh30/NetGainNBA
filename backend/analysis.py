import os
import pandas as pd
import glob
import numpy as np
from collections import Counter, defaultdict

def analyze_simulations():
    """
    analyze all simulation CSV files in the 1k_sims folder and generate statistics
    """
    print("Starting NBA Playoff Simulation Analysis...")
    
    # path to the simulation files
    sims_folder = "data/1k_sims"
    output_file = "data/1k_sims_analysis.csv"
    
    # check if the folder exists
    if not os.path.exists(sims_folder):
        print(f"Error: {sims_folder} directory not found")
        return
    
    # get all CSV files in the folder
    simulation_files = glob.glob(f"{sims_folder}/*.csv")
    
    if not simulation_files:
        print(f"No simulation files found in {sims_folder}")
        return
    
    print(f"Found {len(simulation_files)} simulation files for analysis")
    
    # initialize data structures to store analysis results
    champions = Counter()  # count of championship wins by team
    finals_appearances = Counter()  # count of finals appearances by team
    conf_finals_appearances = Counter()  # count of conference finals appearances
    semifinal_appearances = Counter()  # count of semifinal appearances
    round_progression = defaultdict(lambda: [0, 0, 0, 0, 0])  # track how far teams progress [R1, R2, R3, R4, Champion]
    finals_matchups = Counter()  # track finals matchups
    series_lengths = defaultdict(list)  # track series lengths by round
    upsets = defaultdict(int)  # track upsets (lower seed beating higher seed)
    all_matchups = Counter()  # track all unique matchups across all rounds
    matchup_wins = defaultdict(lambda: {"team1_wins": 0, "team2_wins": 0, "total": 0})  # track wins by each team in matchups
    valid_simulations = 0  # count of valid simulation files processed
    
    # Process each simulation file
    for i, sim_file in enumerate(simulation_files):
        try:
            # Read the simulation data
            df = pd.read_csv(sim_file)
            
            # Check if file has the expected data
            if 'round' not in df.columns or 'series_winner_name' not in df.columns:
                print(f"Warning: File {sim_file} is missing required columns")
                continue
                
            # Filter for series summaries only (not individual games)
            summaries = df[df['record_type'] == 'series_summary']
            
            if summaries.empty:
                print(f"Warning: No series summaries found in {sim_file}")
                continue
            
            # Find all teams in this simulation
            all_teams = set()
            for _, row in summaries.iterrows():
                all_teams.add(row['team_a_name'])
                all_teams.add(row['team_b_name'])
            
            # Track round progression for each team
            team_progress = {team: 0 for team in all_teams}
            
            # Process each round
            for round_num in range(1, 5):  # Rounds 1-4
                round_data = summaries[summaries['round'] == round_num]
                
                # Skip if round data is missing
                if round_data.empty:
                    break
                
                # Process each series in this round
                for _, series in round_data.iterrows():
                    winner = series['series_winner_name']
                    loser = series['team_a_name'] if series['series_winner_name'] == series['team_b_name'] else series['team_b_name']
                    
                    # Update team progress
                    team_progress[winner] = round_num
                    team_progress[loser] = max(team_progress.get(loser, 0), round_num - 1)
                    
                    # Track series length
                    series_length = min(series['team_a_wins'] + series['team_b_wins'], 7)
                    series_lengths[f"Round {round_num}"].append(series_length)
                    
                    # Track all unique matchups (normalize order by sorting team names)
                    '''
                    teams = sorted([series['team_a_name'], series['team_b_name']])
                    matchup_key = f"{teams[0]} vs {teams[1]}"
                    all_matchups[matchup_key] += 1
                    '''

                    if winner == series['team_a_name']:
                        matchup_wins[matchup_key]["team1_wins"] += 1
                    else:
                        matchup_wins[matchup_key]["team2_wins"] += 1

                    
                    # Track who won in this matchup
                    matchup_wins[matchup_key]["total"] += 1
                    if winner == teams[0]:
                        matchup_wins[matchup_key]["team1_wins"] += 1
                    else:
                        matchup_wins[matchup_key]["team2_wins"] += 1

                    # Check for upsets if seed data is available
                    if 'team_a_seed' in series and 'team_b_seed' in series:
                        team_a_seed = series['team_a_seed']
                        team_b_seed = series['team_b_seed']
                        
                        if not pd.isna(team_a_seed) and not pd.isna(team_b_seed):
                            if (team_a_seed > team_b_seed and series['series_winner_name'] == series['team_a_name']) or \
                               (team_b_seed > team_a_seed and series['series_winner_name'] == series['team_b_name']):
                                upsets[f"Round {round_num}"] += 1
                    
                    # For finals (round 4)
                    if round_num == 4:
                        # Record finals matchup
                        matchup = f"{series['team_a_name']} vs {series['team_b_name']}"
                        finals_matchups[matchup] += 1
                        
                        # Record finalists
                        finals_appearances[series['team_a_name']] += 1
                        finals_appearances[series['team_b_name']] += 1
                        
                        # Record champion
                        champions[winner] += 1
                    
                    # For conference finals (round 3)
                    elif round_num == 3:
                        conf_finals_appearances[series['team_a_name']] += 1
                        conf_finals_appearances[series['team_b_name']] += 1
                    
                    # For semifinals (round 2)
                    elif round_num == 2:
                        semifinal_appearances[series['team_a_name']] += 1
                        semifinal_appearances[series['team_b_name']] += 1
            
            # Update round progression counters
            for team, max_round in team_progress.items():
                if max_round > 0:  # Only count teams that participated
                    # Mark all rounds the team participated in
                    for r in range(max_round):
                        round_progression[team][r] += 1
                    
                    # Mark if they were champions
                    if max_round == 4 and team in champions:
                        round_progression[team][4] += 1
            
            valid_simulations += 1
            
            # Show progress
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(simulation_files)} files")
                
        except Exception as e:
            print(f"Error processing {sim_file}: {str(e)}")
    
    print(f"Successfully processed {valid_simulations} valid simulation files")
    
    # Calculate percentages based on valid simulations
    if valid_simulations > 0:
        # Generate champion analysis dataframe
        champions_df = pd.DataFrame({
            'Team': list(champions.keys()),
            'Championships': list(champions.values()),
            'Championship_%': [count / valid_simulations * 100 for count in champions.values()],
            'Finals_Appearances': [finals_appearances[team] for team in champions.keys()],
            'Finals_Appearance_%': [finals_appearances[team] / valid_simulations * 100 for team in champions.keys()],
            'Conf_Finals_Appearances': [conf_finals_appearances[team] for team in champions.keys()],
            'Conf_Finals_%': [conf_finals_appearances[team] / valid_simulations * 100 for team in champions.keys()],
            'Semifinals_Appearances': [semifinal_appearances[team] for team in champions.keys()],
            'Semifinals_%': [semifinal_appearances[team] / valid_simulations * 100 for team in champions.keys()]
        })
        
        # Add teams that never won but made finals
        non_champ_finalists = [team for team in finals_appearances if team not in champions]
        if non_champ_finalists:
            non_champ_df = pd.DataFrame({
                'Team': non_champ_finalists,
                'Championships': [0] * len(non_champ_finalists),
                'Championship_%': [0.0] * len(non_champ_finalists),
                'Finals_Appearances': [finals_appearances[team] for team in non_champ_finalists],
                'Finals_Appearance_%': [finals_appearances[team] / valid_simulations * 100 for team in non_champ_finalists],
                'Conf_Finals_Appearances': [conf_finals_appearances[team] for team in non_champ_finalists],
                'Conf_Finals_%': [conf_finals_appearances[team] / valid_simulations * 100 for team in non_champ_finalists],
                'Semifinals_Appearances': [semifinal_appearances[team] for team in non_champ_finalists],
                'Semifinals_%': [semifinal_appearances[team] / valid_simulations * 100 for team in non_champ_finalists]
            })
            champions_df = pd.concat([champions_df, non_champ_df])
        
        # Add teams that never made finals but made conference finals
        non_finalist_conf = [team for team in conf_finals_appearances if team not in champions and team not in non_champ_finalists]
        if non_finalist_conf:
            non_finalist_df = pd.DataFrame({
                'Team': non_finalist_conf,
                'Championships': [0] * len(non_finalist_conf),
                'Championship_%': [0.0] * len(non_finalist_conf),
                'Finals_Appearances': [0] * len(non_finalist_conf),
                'Finals_Appearance_%': [0.0] * len(non_finalist_conf),
                'Conf_Finals_Appearances': [conf_finals_appearances[team] for team in non_finalist_conf],
                'Conf_Finals_%': [conf_finals_appearances[team] / valid_simulations * 100 for team in non_finalist_conf],
                'Semifinals_Appearances': [semifinal_appearances[team] for team in non_finalist_conf],
                'Semifinals_%': [semifinal_appearances[team] / valid_simulations * 100 for team in non_finalist_conf]
            })
            champions_df = pd.concat([champions_df, non_finalist_df])
        
        # Sort by championships, then finals appearances
        champions_df = champions_df.sort_values(by=['Championships', 'Finals_Appearances', 'Conf_Finals_Appearances'], 
                                              ascending=False).reset_index(drop=True)
        
        # Round percentages to 2 decimal places
        pct_columns = [col for col in champions_df.columns if '_%' in col]
        champions_df[pct_columns] = champions_df[pct_columns].round(2)
        
        # Generate finals matchups analysis
        matchups_df = pd.DataFrame({
            'Matchup': list(finals_matchups.keys()),
            'Count': list(finals_matchups.values()),
            'Percentage': [count / valid_simulations * 100 for count in finals_matchups.values()]
        }).sort_values(by='Count', ascending=False).reset_index(drop=True)
        
        matchups_df['Percentage'] = matchups_df['Percentage'].round(2)
        
        # Generate all unique matchups analysis
        all_matchups_data = []
        for matchup, count in all_matchups.items():
            team1, team2 = matchup.split(" vs ")
            team1_wins = matchup_wins[matchup]["team1_wins"]
            team2_wins = matchup_wins[matchup]["team2_wins"]
            total = matchup_wins[matchup]["total"]
            
            all_matchups_data.append({
                'Matchup': matchup,
                'Occurrences': count,
                'Percentage': (count / valid_simulations * 100),
                f'{team1}_Win_%': (team1_wins / total * 100) if total > 0 else 0,
                f'{team2}_Win_%': (team2_wins / total * 100) if total > 0 else 0
            })
            
        all_matchups_df = pd.DataFrame(all_matchups_data).sort_values(by='Occurrences', ascending=False).reset_index(drop=True)
        
        # Round percentages to 2 decimal places
        pct_columns = [col for col in all_matchups_df.columns if '%' in col]
        all_matchups_df[pct_columns] = all_matchups_df[pct_columns].round(2)
        
        # Generate series length analysis
        series_length_df = pd.DataFrame({
            'Round': [],
            'Avg_Games': [],
            'Most_Common_Length': []
        })
        
        for round_name, lengths in series_lengths.items():
            if lengths:
                avg_length = np.mean(lengths)
                most_common = Counter(lengths).most_common(1)[0][0]
                series_length_df = pd.concat([series_length_df, pd.DataFrame({
                    'Round': [round_name],
                    'Avg_Games': [round(avg_length, 2)],
                    'Most_Common_Length': [most_common]
                })])
        
        # Generate full progression analysis
        progression_df = pd.DataFrame(columns=['Team', 'Round_1', 'Round_2', 'Round_3', 'Finals', 'Champion'])
        
        for team, rounds in round_progression.items():
            progression_df = pd.concat([progression_df, pd.DataFrame({
                'Team': [team],
                'Round_1': [rounds[0]],
                'Round_2': [rounds[1]], 
                'Round_3': [rounds[2]],
                'Finals': [rounds[3]],
                'Champion': [rounds[4]]
            })])
        
        # Calculate percentages for each round
        for col in ['Round_1', 'Round_2', 'Round_3', 'Finals', 'Champion']:
            # Convert the column to numeric first to handle any non-numeric values
            progression_df[col] = pd.to_numeric(progression_df[col], errors='coerce').fillna(0)
            # Calculate percentage and round
            progression_df[f'{col}_%'] = ((progression_df[col] / valid_simulations) * 100).round(2)
        
        progression_df = progression_df.sort_values(by='Champion', ascending=False).reset_index(drop=True)
        
        # Save all analysis to the output file
        # Create a writer for multiple sheets
        os.makedirs('data', exist_ok=True)  # Ensure data directory exists
        
        try:
            with pd.ExcelWriter("data/1k_sims_analysis.xlsx") as writer:
                champions_df.to_excel(writer, sheet_name='Team_Analysis', index=False)
                matchups_df.to_excel(writer, sheet_name='Finals_Matchups', index=False)
                all_matchups_df.to_excel(writer, sheet_name='All_Matchups', index=False)
                series_length_df.to_excel(writer, sheet_name='Series_Length', index=False)
                progression_df.to_excel(writer, sheet_name='Round_Progression', index=False)
            
            # Also save the main analysis as CSV
            champions_df.to_csv(output_file, index=False)
            # Save all unique matchups analysis as CSV
            all_matchups_df.to_csv("data/1k_sims_unique_matchups.csv", index=False)
            
            print(f"Analysis complete. Results saved to data/1k_sims_analysis.xlsx and {output_file}")
            print(f"All unique matchups saved to data/1k_sims_unique_matchups.csv")
        except Exception as e:
            print(f"Error saving Excel file: {str(e)}")
            # Fallback to CSV only if Excel fails
            champions_df.to_csv(output_file, index=False)
            matchups_df.to_csv("data/1k_sims_finals_matchups.csv", index=False)
            all_matchups_df.to_csv("data/1k_sims_unique_matchups.csv", index=False)
            series_length_df.to_csv("data/1k_sims_series_length.csv", index=False)
            progression_df.to_csv("data/1k_sims_progression.csv", index=False)
            print(f"Analysis complete. Results saved as CSV files in the data directory")
        
        # Return the top champion
        if not champions_df.empty:
            top_champion = champions_df.iloc[0]
            print(f"\nMost common champion: {top_champion['Team']} won {top_champion['Championships']} times ({top_champion['Championship_%']}%)")
            
            # Return the most common finals matchup
            if not matchups_df.empty:
                top_matchup = matchups_df.iloc[0]
                print(f"Most common finals matchup: {top_matchup['Matchup']} occurred {top_matchup['Count']} times ({top_matchup['Percentage']}%)")
                
            # Return the most common overall matchup
            if not all_matchups_df.empty:
                top_overall_matchup = all_matchups_df.iloc[0]
                print(f"Most common overall matchup: {top_overall_matchup['Matchup']} occurred {top_overall_matchup['Occurrences']} times ({top_overall_matchup['Percentage']}%)")
    else:
        print("No valid simulations were processed. Check the file format and contents.")

if __name__ == "__main__":
    analyze_simulations()
