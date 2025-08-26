from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import json
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import random
import analysis
import numpy as np  # add numpy import for advanced distributions

app = Flask(__name__)
cors = CORS(app, origins="http://localhost:5173")

# load team mapping from json file
def load_team_mapping():
    try:
        with open("team_mapping.json", "r") as f:
            # convert string keys to integers
            return {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"error loading team mapping: {str(e)}")
        return {}

TEAM_ID_TO_NAME = load_team_mapping()

@app.post("/get_matchup_stats")
def get_matchup_stats():
    # parse the json payload from the request body
    data = request.get_json()
    team_a_id = int(data.get("team_a_id"))
    team_a_szn = data.get("team_a_szn")
    team_b_id = int(data.get("team_b_id"))
    team_b_szn = data.get("team_b_szn")
    
    # normalize season format (e.g., "2021-2022" to "2021-22")
    team_a_szn = normalize_season_format(team_a_szn)
    team_b_szn = normalize_season_format(team_b_szn)

    try:
        # path to the dataset and output files
        dataset_path = "data/TEAMS_CLUSTERED.csv"
        output_path = "data/matchup_stats.csv"
        team_output_path = "data/matchup_team_id.csv"
        
        # get round number from request data, default to 1
        round_number = data.get("round", 1)
        
        # read dataset
        df = pd.read_csv(dataset_path)
        
        # filter rows for the specific teams and seasons
        team_a_row = df[(df["TEAM_ID"] == team_a_id) & (df["SEASON"] == team_a_szn)]
        team_b_row = df[(df["TEAM_ID"] == team_b_id) & (df["SEASON"] == team_b_szn)]
        
        if team_a_row.empty or team_b_row.empty:
            print("no data found for the specified teams and seasons.")
            return jsonify({
                "status": "error",
                "message": "Invalid team or season provided."
            }), 400
        
        # drop the unwanted columns
        columns_to_drop = ["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "CLUSTER"]
        team_a_row = team_a_row.drop(columns=columns_to_drop, errors='ignore')
        team_b_row = team_b_row.drop(columns=columns_to_drop, errors='ignore')
        
        # compute difference of rows
        diff_df = team_a_row.reset_index(drop=True) - team_b_row.reset_index(drop=True)
        
        # add the round column
        diff_df['ROUND'] = round_number
        
        # save output file
        save_to_csv(diff_df, output_path)
        
        # create a dataframe with team IDs
        team_id_df = pd.DataFrame({
            'team_a_id': [team_a_id],
            'team_b_id': [team_b_id],
            'round': [round_number]
        })
        
        # save team ID output file
        save_to_csv(team_id_df, team_output_path)
        
        return jsonify({
            "status": "success",
            "message": f"Added differential stats for {team_a_id} ({team_a_szn}) vs {team_b_id} ({team_b_szn}) for round {round_number}"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.get("/determine_winners")
def determine_winners():
    try:
        csv_path = "data/matchup_stats.csv"
        team_id_path = "data/matchup_team_id.csv"
        simulation_path = "data/simulation_data.csv"
        
        # check if reset parameter is provided
        reset = request.args.get('reset', 'false').lower() == 'true'
        if reset:
            # reset files if they exist
            for file_path in [csv_path, team_id_path, simulation_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return jsonify({
                "status": "success",
                "message": "Matchup data has been reset."
            })
        
        # handle multi-round logic
        current_round = 1
        max_rounds = 4  # first round, semifinals, conference finals, finals
        all_series_results = []
        
        # check if we're continuing from a previous round
        if os.path.exists(simulation_path):
            try:
                existing_sim_data = pd.read_csv(simulation_path)
                if not existing_sim_data.empty and 'round' in existing_sim_data.columns:
                    current_round = existing_sim_data['round'].max() + 1
                    if current_round > max_rounds:
                        return jsonify({
                            "status": "success", 
                            "message": "Tournament is already complete",
                            "final_results": "Check simulation_data.csv for full tournament results"
                        })
            except Exception as e:
                print(f"error reading existing simulation data: {str(e)}")
                current_round = 1
        
        print(f"Processing playoff round: {current_round}")
        
        # process each round until we have a champion
        while current_round <= max_rounds:
            # check if we have matchups for the current round
            if not os.path.exists(team_id_path) or not os.path.exists(csv_path):
                if current_round == 1:
                    return jsonify({
                        "status": "error",
                        "message": "CSV files not found. Run get_matchup_stats first."
                    }), 404
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Data files for round {current_round} are missing."
                    }), 500
            
            # read both CSV files and filter for current round
            team_id_df = pd.read_csv(team_id_path)
            
            # filter by round if the column exists
            if 'round' in team_id_df.columns:
                current_round_matchups = team_id_df[team_id_df['round'] == current_round]
                if current_round_matchups.empty:
                    return jsonify({
                        "status": "error",
                        "message": f"No matchups found for round {current_round}."
                    }), 404
            else:
                # if no round column exists, we're likely in round 1 with old data
                current_round_matchups = team_id_df
            
            # read the matchup stats and filter similarly
            df = pd.read_csv(csv_path)
            
            # if ROUND column exists, filter by it
            if 'ROUND' in df.columns:
                current_stats = df[df['ROUND'] == current_round].drop(columns=['ROUND'])
            else:
                # handle case with old data without ROUND column
                if len(df) >= len(current_round_matchups):
                    current_stats = df.iloc[:len(current_round_matchups)].copy()
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Stats data does not match matchup data."
                    }), 500
            
            model1_config = load_model_config("nba_tabnet_model_config.json")
            clf1 = initialize_model(model1_config, "NBA_TABNET_TEST_MODEL.zip")

            model2_config = load_model_config("nba_tabnet_model_2_config.json")
            clf2 = initialize_model(model2_config, "NBA_TABNET_TEST_MODEL_2.zip")

            # get class predictions and probabilities
            '''
            predictions = clf.predict(current_stats.values)
            probabilities = clf.predict_proba(current_stats.values)
            '''

            # Regular order predictions (Team A - Team B)
            probs1 = clf1.predict_proba(current_stats.values)
            probs2 = clf2.predict_proba(current_stats.values)
            blended_regular = (probs1 + probs2) / 2

            # Flipped input predictions (Team B - Team A)
            flipped_stats = -current_stats.values
            probs1_flipped = clf1.predict_proba(flipped_stats)
            probs2_flipped = clf2.predict_proba(flipped_stats)
            blended_flipped = (probs1_flipped + probs2_flipped) / 2

            # Average the positive class probabilities from both perspectives
            positive_probs_regular = blended_regular[:, 1]
            positive_probs_flipped = 1 - blended_flipped[:, 1]  # because flipping reverses win prob

            # Final symmetric probabilities
            final_probs = (positive_probs_regular + positive_probs_flipped) / 2

            # Create 2D probability array for consistency with downstream logic
            probabilities = np.stack([1 - final_probs, final_probs], axis=1)
            predictions = (final_probs >= 0.505).astype(int)
            
            # organize matchups for this round
            matchups = organize_matchups(current_round_matchups, predictions, probabilities)
            
            # simulate the best-of-7 series for each matchup
            series_results, next_round_winners = simulate_playoff_series(matchups, current_round)
            all_series_results.extend(series_results)
            
            # save the simulation results to CSV
            save_simulation_results(series_results, simulation_path, reset)
            
            # set up next round matchups if not the final round
            if current_round < max_rounds and next_round_winners:
                # create matchups for the next round
                next_round = current_round + 1
                next_matchups = create_next_round_matchups(next_round_winners, next_round)
                
                # set up data for each new matchup
                for matchup in next_matchups:
                    setup_next_round_matchup(matchup, next_round)
                
                # move to the next round
                current_round = next_round
            else:
                # if we're in the final round or have unexpected number of winners, stop here
                break
        
        # return results of all rounds
        return jsonify({
            "status": "success",
            "message": f"Completed simulation through round {current_round}",
            "series_results": all_series_results,
            "diagnostics": {
                "total_rounds_processed": current_round,
                "total_series_simulated": len(all_series_results)
            }
        })
        
    except Exception as e:
        print("error in determine_winners endpoint:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# helper functions
def normalize_season_format(season):
    """normalize season format (e.g., "2021-2022" to "2021-22")"""
    season = str(season)
    if "-" in season and len(season.split("-")) == 2:
        start, end = season.split("-")
        return f"{start}-{end[-2:]}"
    return season

def save_to_csv(df, file_path):
    """save dataframe to csv, creating file if it doesn't exist"""
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        # check if header needs to be included
        include_header = not os.path.getsize(file_path) > 0
        df.to_csv(file_path, mode='a', header=include_header, index=False)

def load_model_config(config_path):
    """load model configuration from json file"""
    with open(config_path, "r") as f:
        return json.load(f)
    
# initialize the tabnet model with configuration
def initialize_model(model_config, model_load):
    
    # use CUDA if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for model inference")
    
    clf = TabNetClassifier(
        **model_config["best_params"],
        optimizer_params=model_config["optimizer_params"],
        seed=model_config["seed"],
        verbose=0,
        device_name=device
    )
    return clf

def organize_matchups(matchups_df, predictions, probabilities):
    """organize matchup data with predictions"""
    matchups = {}
    for i, (pred, prob_arr) in enumerate(zip(predictions, probabilities)):
        if i >= len(matchups_df):
            break
            
        team_a_id = int(matchups_df.iloc[i]['team_a_id'])
        team_b_id = int(matchups_df.iloc[i]['team_b_id'])
        team_a_prob = float(prob_arr[1])
        
        matchup_key = (team_a_id, team_b_id)
        
        if matchup_key not in matchups:
            matchups[matchup_key] = {
                'team_a_id': team_a_id,
                'team_a_name': TEAM_ID_TO_NAME.get(team_a_id, "Unknown Team"),
                'team_b_id': team_b_id,
                'team_b_name': TEAM_ID_TO_NAME.get(team_b_id, "Unknown Team"),
                'win_probability': team_a_prob,
                'round': matchups_df.iloc[i].get('round', 1),
                'series_results': []
            }
    
    return matchups

def simulate_playoff_series(matchups, round_num):
    """simulate best-of-7 series for all matchups"""
    series_results = []
    next_round_winners = []
    
    for matchup_key, matchup_info in matchups.items():
        team_a_id = matchup_info['team_a_id']
        team_a_name = matchup_info['team_a_name']
        team_b_id = matchup_info['team_b_id']
        team_b_name = matchup_info['team_b_name']
        win_probability = matchup_info['win_probability']
        
        # initialize series tracking
        team_a_wins = 0
        team_b_wins = 0
        games = []
        
        print(f"\nSimulating round {round_num} series: {team_a_name} vs {team_b_name}")
        print(f"Model predicts {team_a_name} has a {win_probability:.2%} chance of winning the series")
        
        # simulate games until one team reaches 4 wins
        for game_num in range(1, 8):
            if team_a_wins >= 4 or team_b_wins >= 4:
                break

            is_team_a_home = game_num in [1, 2, 5, 7]  # assuming team A has home court advantage
            home_team_id = team_a_id if is_team_a_home else team_b_id # Define home_team_id here
            home_boost = 0.05 if is_team_a_home else -0.05
            adjusted_prob = np.clip(win_probability + home_boost, 0.05, 0.95) # Define adjusted_prob here

            momentum_winner_id = None
            if len(games) >= 2 and games[-1]['winner_id'] == games[-2]['winner_id']:
                momentum_winner_id = games[-1]['winner_id']
                momentum_boost = 0.03
                adjusted_prob += momentum_boost if momentum_winner_id == team_a_id else -momentum_boost # Update adjusted_prob
            else:
                momentum_boost = 0.0  # for logging

            # Clip adjusted prob before sampling
            adjusted_prob = np.clip(adjusted_prob, 0.05, 0.95) # Clip adjusted_prob

            alpha = adjusted_prob * 3 # use adjusted_prob
            beta = (1 - adjusted_prob) * 3 # use adjusted_prob
            
            # sample from Beta distribution to get a randomized win probability
            beta_sample = np.random.beta(alpha, beta)
            
            # add noise factor to occasionally cause upsets
            # this stretches or compresses probabilities to increase upset potential
            noise_factor = np.random.normal(1.0, 0.15)  # normal distribution centered at 1.0 with std dev 0.15
            modified_prob = np.clip(beta_sample * noise_factor, 0.1, 0.9)  # clip to prevent extreme values
            
            # use modified Bernoulli trial with the adjusted probability
            random_value = np.random.random()  # uniform for comparison
            
            # determine winner based on sampled probability
            if random_value < modified_prob:
                winner_id = team_a_id
                winner_name = team_a_name
                team_a_wins += 1
            else:
                winner_id = team_b_id
                winner_name = team_b_name
                team_b_wins += 1
            
            # record game result
            game_result = {
                "game_number": game_num,
                "random_value": float(random_value),
                "beta_sample": float(modified_prob),  # store the modified probability
                "noise_factor": float(noise_factor),  # store the noise factor for reference
                "winner_id": winner_id,
                "winner_name": winner_name,
                "home_boost_applied": float(home_boost if home_team_id == team_a_id else -home_boost),
                "momentum_boost_applied": float(momentum_boost if momentum_winner_id == team_a_id else (-momentum_boost if momentum_winner_id else 0)),
            }
            games.append(game_result)
            
            print(f"Game {game_num}: Beta={beta_sample:.4f}, Noise={noise_factor:.4f}, Random={random_value:.4f} - Winner: {winner_name}")
        
        # determine series winner
        if team_a_wins > team_b_wins:
            series_winner_id = team_a_id
            series_winner_name = team_a_name
        else:
            series_winner_id = team_b_id
            series_winner_name = team_b_name
        
        # add winner to next round matchups
        next_round_winners.append({
            'team_id': series_winner_id,
            'team_name': series_winner_name
        })
        
        # create series summary
        series_summary = {
            "team_a_id": int(team_a_id),
            "team_a_name": team_a_name,
            "team_b_id": int(team_b_id),
            "team_b_name": team_b_name,
            "win_probability": float(win_probability),
            "team_a_wins": int(team_a_wins),
            "team_b_wins": int(team_b_wins),
            "series_winner_id": int(series_winner_id),
            "series_winner_name": series_winner_name,
            "round": int(round_num),
            "games": games
            
        }
        
        series_results.append(series_summary)
        
        # print series summary
        print(f"Series result: {team_a_name} {team_a_wins}-{team_b_wins} {team_b_name}")
        print(f"Series winner: {series_winner_name}")
        print("-" * 50)
    
    return series_results, next_round_winners

def save_simulation_results(series_results, simulation_path, reset=False):
    """save simulation results to CSV file"""
    try:
        # create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # prepare data for CSV - create flattened records
        csv_records = []
        
        for idx, series in enumerate(series_results):
            # add series-level summary
            series_record = {
                "simulation_id": idx + 1,
                "record_type": "series_summary",
                "team_a_id": series["team_a_id"],
                "team_a_name": series["team_a_name"],
                "team_b_id": series["team_b_id"],
                "team_b_name": series["team_b_name"],
                "win_probability": series["win_probability"],
                "team_a_wins": series["team_a_wins"],
                "team_b_wins": series["team_b_wins"],
                "series_winner_id": series["series_winner_id"],
                "series_winner_name": series["series_winner_name"],
                "total_games": len(series["games"]),
                "game_number": None,
                "game_win_probability": None,
                "game_winner_id": None,
                "game_winner_name": None,
                "round": series["round"]
            }
            csv_records.append(series_record)
            
            # add individual game records
            for game in series["games"]:
                game_record = {
                    "simulation_id": idx + 1,
                    "record_type": "game",
                    "team_a_id": series["team_a_id"],
                    "team_a_name": series["team_a_name"],
                    "team_b_id": series["team_b_id"],
                    "team_b_name": series["team_b_name"],
                    "win_probability": series["win_probability"],
                    "team_a_wins": None,
                    "team_b_wins": None,
                    "series_winner_id": None,
                    "series_winner_name": None,
                    "total_games": None,
                    "game_number": game["game_number"],
                    "game_win_probability": game["random_value"],
                    "game_winner_id": int(game["winner_id"]),
                    "game_winner_name": game["winner_name"],
                    "round": series["round"]
                }
                csv_records.append(game_record)
        
        # convert to dataframe
        simulation_df = pd.DataFrame(csv_records)
        
        # convert team id columns to Int64
        int_columns = ['team_a_id', 'team_b_id', 'series_winner_id', 'game_winner_id']
        for col in int_columns:
            simulation_df[col] = pd.Series(simulation_df[col]).astype('Int64')
        
        # append to existing file or create new one
        if os.path.exists(simulation_path) and not reset:
            include_header = not os.path.getsize(simulation_path) > 0
            simulation_df.to_csv(simulation_path, mode='a', header=include_header, index=False)
        else:
            simulation_df.to_csv(simulation_path, index=False)
        
        print(f"saved round {series_results[0]['round']} simulation results to data/simulation_data.csv")
        
    except Exception as e:
        print(f"warning: could not save simulation data to CSV: {str(e)}")

def simulate_tournament(model=None, model2=None, threshold = 0.505):
    """simulate the tournament by running all rounds; returns a message string"""
    simulation_path = "data/simulation_data.csv"
    csv_path = "data/matchup_stats.csv"
    team_id_path = "data/matchup_team_id.csv"
    # if simulation file exists, remove it to start fresh
    if os.path.exists(simulation_path):
        os.remove(simulation_path)
    current_round = 1
    max_rounds = 4  # first round, semifinals, conference finals, finals
    all_series_results = []
    
    # if model is not provided, load it once outside the loop
    if model is None and model2 is None:
        model_config = load_model_config("nba_tabnet_model_config.json")
        clf = initialize_model(model_config, "NBA_TABNET_TEST_MODEL.zip")
        model2_config = load_model_config("nba_tabnet_model_2_config.json")
        clf2 = initialize_model(model2_config, "NBA_TABNET_TEST_MODEL_2.zip")
    
    while current_round <= max_rounds:
        # check if matchup files exist
        if not os.path.exists(team_id_path) or not os.path.exists(csv_path):
            raise Exception("csv files not found. run get_matchup_stats first.")
        team_id_df = pd.read_csv(team_id_path)
        # filter matchups by round if possible
        if 'round' in team_id_df.columns:
            current_round_matchups = team_id_df[team_id_df['round'] == current_round]
            if current_round_matchups.empty:
                raise Exception(f"no matchups found for round {current_round}.")
        else:
            current_round_matchups = team_id_df
        df = pd.read_csv(csv_path)
        # filter stats by round if possible
        if 'ROUND' in df.columns:
            current_stats = df[df['ROUND'] == current_round].drop(columns=['ROUND'])
        else:
            if len(df) >= len(current_round_matchups):
                current_stats = df.iloc[:len(current_round_matchups)].copy()
            else:
                raise Exception("stats data does not match matchup data.")
        
        flipped_stats = -current_stats.values
        probs1 = clf.predict_proba(current_stats.values)
        probs2 = clf2.predict_proba(current_stats.values)
        blended_forward = (probs1 + probs2) / 2

        flipped_probs1 = clf.predict_proba(flipped_stats)
        flipped_probs2 = clf2.predict_proba(flipped_stats)
        blended_flipped = (flipped_probs1 + flipped_probs2) / 2

        symmetric_blended = (blended_forward + (1 - blended_flipped[:, ::-1])) / 2

        positive_probs = symmetric_blended[:, 1]
        predictions = (positive_probs >= threshold).astype(int)
        probabilities = symmetric_blended

        '''
        # use the pre-loaded model to make predictions and get probabilities
        predictions = clf.predict(current_stats.values)
        probabilities = clf.predict_proba(current_stats.values)
        '''
        
        matchups = organize_matchups(current_round_matchups, predictions, probabilities)
        series_results, next_round_winners = simulate_playoff_series(matchups, current_round)
        all_series_results.extend(series_results)
        save_simulation_results(series_results, simulation_path, reset=False)
        if current_round < max_rounds and next_round_winners:
            next_round = current_round + 1
            next_matchups = create_next_round_matchups(next_round_winners, next_round)
            for matchup in next_matchups:
                setup_next_round_matchup(matchup, next_round)
            current_round = next_round
        else:
            break
    return f"completed simulation through round {current_round}"

def create_next_round_matchups(winners, next_round):
    """create matchups for the next playoff round"""
    if next_round == 2:  # conference semifinals (4 matchups)
        # pair winners: 0 vs 1, 2 vs 3, etc.
        next_matchups = []
        for i in range(0, len(winners), 2):
            if i+1 < len(winners):
                next_matchups.append((winners[i], winners[i+1]))
        return next_matchups
    
    elif next_round == 3:  # conference finals (2 matchups)
        # pair the 4 semifinal winners: 0 vs 1, 2 vs 3
        next_matchups = []
        for i in range(0, len(winners), 2):
            if i+1 < len(winners):
                next_matchups.append((winners[i], winners[i+1]))
        return next_matchups
    
    elif next_round == 4:  # nba finals (1 matchup)
        # pair the 2 conference winners
        if len(winners) >= 2:
            return [(winners[0], winners[1])]
        return []
    
    return []

def setup_next_round_matchup(matchup, next_round):
    """set up a matchup for the next playoff round"""
    team_a = matchup[0]
    team_b = matchup[1]
    
    # get team data from the dataset
    teams_df = pd.read_csv("data/TEAMS_CLUSTERED.csv")
    
    # get the most recent season for each team
    team_a_data = teams_df[teams_df["TEAM_ID"] == team_a['team_id']].sort_values(by="SEASON", ascending=False)
    team_b_data = teams_df[teams_df["TEAM_ID"] == team_b['team_id']].sort_values(by="SEASON", ascending=False)
    
    if not team_a_data.empty and not team_b_data.empty:
        team_a_szn = team_a_data.iloc[0]["SEASON"]
        team_b_szn = team_b_data.iloc[0]["SEASON"]
        
        # prepare data for get_matchup_stats
        matchup_data = {
            "team_a_id": int(team_a['team_id']),
            "team_a_szn": team_a_szn,
            "team_b_id": int(team_b['team_id']),
            "team_b_szn": team_b_szn,
            "round": next_round
        }
        
        # use internal call to get_matchup_stats
        with app.test_client() as client:
            response = client.post('/get_matchup_stats', json=matchup_data)
            if response.status_code != 200:
                print(f"error setting up matchup for round {next_round}: {response.get_json()}")

@app.post("/simulate_multiple")
def simulate_multiple():
    # parse json payload
    data = request.get_json()
    simulate_flag = data.get("simulate_1000", False)
    if not simulate_flag:
        return jsonify({"status": "success", "message": "simulate_1000 flag is false. run simulation as normal using /determine_winners endpoint."})
    
    sims_dir = "data/1k_sims"
    # create directory if it does not exist
    os.makedirs(sims_dir, exist_ok=True)
    results_summary = []
    
    '''
    # preload the model once for all simulation runs
    model_config = load_model_config("nba_tabnet_model_config.json")
    shared_model = initialize_model(model_config)
    '''

    model1_config = load_model_config("nba_tabnet_model_config.json")
    model2_config = load_model_config("nba_tabnet_model_2_config.json")

    clf1 = initialize_model(model1_config, "NBA_TABNET_TEST_MODEL.zip")
    clf2 = initialize_model(model2_config, "NBA_TABNET_TEST_MODEL_2.zip")
    
    # make sure required input files exist before starting the simulations
    csv_path = "data/matchup_stats.csv"
    team_id_path = "data/matchup_team_id.csv"
    if not os.path.exists(csv_path) or not os.path.exists(team_id_path):
        return jsonify({
            "status": "error",
            "message": "Required input files (matchup_stats.csv or matchup_team_id.csv) are missing. Please run get_matchup_stats first."
        }), 400
    
    # create temp copies of the original matchup files
    temp_csv_path = "data/temp_matchup_stats.csv"
    temp_team_id_path = "data/temp_matchup_team_id.csv"
    
    try:
        # create one-time backup of original files
        import shutil
        shutil.copy2(csv_path, temp_csv_path)
        shutil.copy2(team_id_path, temp_team_id_path)
        
        for i in range(1, 1001):
            try:
                print(f"{i}/1000")
                
                # restore original state from temp files
                shutil.copy2(temp_csv_path, csv_path)
                shutil.copy2(temp_team_id_path, team_id_path)
                
                # clear any existing simulation_data.csv
                simulation_path = "data/simulation_data.csv"
                if os.path.exists(simulation_path):
                    os.remove(simulation_path)
                
                # run a full tournament simulation using the shared model
                msg = simulate_tournament(model=shared_model)
                
                # check if simulation data was created
                if os.path.exists(simulation_path):
                    dest_file = os.path.join(sims_dir, f"{i}_simulation_data.csv")
                    try:
                        shutil.copy2(simulation_path, dest_file)
                        os.remove(simulation_path)  # remove after copying
                        print(f"Successfully saved simulation {i} to {dest_file}")
                    except Exception as copy_error:
                        print(f"Error copying file: {str(copy_error)}")
                        raise copy_error
                else:
                    error_msg = f"simulation_data.csv not found after simulation run {i}"
                    print(error_msg)
                    raise Exception(error_msg)
                    
                results_summary.append({"simulation_run": i, "output_file": f"{i}_simulation_data.csv", "message": msg})
            except Exception as e:
                error_message = f"Error in simulation {i}: {str(e)}"
                print(error_message)
                results_summary.append({"simulation_run": i, "error": error_message})
    
    finally:
        # clean up temp files at the end
        for file_path in [temp_csv_path, temp_team_id_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {file_path}: {str(e)}")
    
    # count successful vs failed simulations
    successful = sum(1 for result in results_summary if "error" not in result)
    failed = len(results_summary) - successful
    
    # check other python scripts for analysis
    analysis.analyze_simulations()
    
    return jsonify({
        "status": "success", 
        "message": f"Completed {successful} simulations successfully, {failed} failed",
        "results": results_summary
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)