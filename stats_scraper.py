import numpy as np
import pandas as pd
import time
import os

#nba api
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.endpoints import leaguedashplayershotlocations
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguedashteamshotlocations
from nba_api.stats.endpoints import synergyplaytypes

# Record the start time
start_time = time.time()

def year_iterator():
    year_1_min = 1996
    year_2_max = 2024

    year_range = range(year_1_min,year_2_max)

    for y in year_range:
        year = str(y)

        next_year = y + 1  # Increment the year by 1
        next_year_last_two = str(next_year)[-2:]  # Extract the last two digits of the next year

        # Handle the case where the next year's last two digits are '00' (i.e., the year 2000, 2100, etc.)
        if next_year_last_two == '00':
            next_year_last_two = '00'

        yield f"{year}-{next_year_last_two}"

def year_iterator_2():
    year_1_min = 2012
    year_2_max = 2024

    year_range = range(year_1_min,year_2_max)

    for y in year_range:
        year = str(y)

        next_year = y + 1  # Increment the year by 1
        next_year_last_two = str(next_year)[-2:]  # Extract the last two digits of the next year

        # Handle the case where the next year's last two digits are '00' (i.e., the year 2000, 2100, etc.)
        if next_year_last_two == '00':
            next_year_last_two = '00'

        yield f"{year}-{next_year_last_two}"

def player_base_scraper():
    for year in year_iterator():
        print(f"Fetching player scoring data for {year}...")
        season_year = year
        season_type ="Regular Season"

        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season_year,
                                                                season_type_all_star=season_type,
                                                                    per_mode_detailed='PerGame',
                                                                    measure_type_detailed_defense='Base')

        player_stats = player_stats.get_data_frames()[0]
        player_stats_df = pd.DataFrame(player_stats)

        if player_stats_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue
        
        df = player_stats_df
        df['SEASON'] = year #add season column to df

        df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]

        yield df

def player_scoring_scraper():
    for year in year_iterator():
        print(f"Fetching player scoring data for {year}...")
        season_year = year
        season_type ="Regular Season"

        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season_year,
                                                                season_type_all_star=season_type,
                                                                    per_mode_detailed='PerGame',
                                                                    measure_type_detailed_defense='Scoring')

        player_stats = player_stats.get_data_frames()[0]
        player_stats_df = pd.DataFrame(player_stats)

        if player_stats_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue
        
        df = player_stats_df
        df['SEASON'] = year #add season column to df

        df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]

        yield df

def team_scoring_scraper():
    for year in year_iterator():
        print(f"Fetching team scoring data for {year}...")
        season_year = year
        season_type ="Regular Season"

        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season_year,
                                                                season_type_all_star=season_type,
                                                                    per_mode_detailed='PerGame',
                                                                    measure_type_detailed_defense='Scoring')

        team_stats = team_stats.get_data_frames()[0]
        team_stats_df = pd.DataFrame(team_stats)

        if team_stats_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue
        
        df = team_stats_df
        df['SEASON'] = year #add season column to df

        df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]

        yield df

def player_shots_data_scraper():
    for year in year_iterator():
        print(f"Fetching player shots data for {year}...")
        season_year = year
        season_type = "Regular Season"

        shots_data = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(season=season_year,
                                                                                 season_type_all_star=season_type,
                                                                                 per_mode_detailed='PerGame')
        shots_data = shots_data.get_data_frames()[0]
        shots_data_df = pd.DataFrame(shots_data)

        if shots_data_df.empty:
            print(f"Skipping {season_year}: No data available.")
            continue

        # Flatten MultiIndex columns (if any)
        if isinstance(shots_data_df.columns, pd.MultiIndex):
            shots_data_df.columns = [' '.join(col).strip().upper() for col in shots_data_df.columns.values]    

        shots_data_df.columns = [col.replace(" ", "_") for col in shots_data_df.columns]

        df = shots_data_df.copy()
        
        # Fill NaN values with 0 before adding SEASON
        df.fillna(0, inplace=True)

        # Add SEASON column to the dataframe
        df['SEASON'] = season_year

        # Reorder columns to place 'SEASON' at the start
        df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]


        yield df

def team_shots_data_scraper():
    for year in year_iterator():
        print(f"Fetching team shots data for {year}...")
        season_year = year
        season_type = "Regular Season"

        shots_data = leaguedashteamshotlocations.LeagueDashTeamShotLocations(season=season_year,
                                                                                 season_type_all_star=season_type,
                                                                                 per_mode_detailed='PerGame')
        shots_data = shots_data.get_data_frames()[0]
        shots_data_df = pd.DataFrame(shots_data)

        if shots_data_df.empty:
            print(f"Skipping {season_year}: No data available.")
            continue

        # Flatten MultiIndex columns (if any)
        if isinstance(shots_data_df.columns, pd.MultiIndex):
            shots_data_df.columns = [' '.join(col).strip().upper() for col in shots_data_df.columns.values]    

        shots_data_df.columns = [col.replace(" ", "_") for col in shots_data_df.columns] 

        df = shots_data_df
        # Add SEASON column to the dataframe
        df['SEASON'] = season_year

        # Reorder columns to place 'SEASON' at the start
        df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]

        yield df

def player_offensive_playtype_data_scraper():
    for year in year_iterator_2():
        
        playtype_list = ['Isolation', 'Transition', 'PRBallHandler', 'PRRollMan', 'Postup', 'Spotup', 'Handoff',
                        'Cut', 'OffScreen', 'OffRebound']

        for playtype in playtype_list:
            
            print(f"Fetching {playtype} data for {year}...")
            season_year = year
            season_type = "Regular Season"

            playtype_data = synergyplaytypes.SynergyPlayTypes(season=season_year,
                                                        season_type_all_star=season_type,
                                                        per_mode_simple='PerGame',
                                                        play_type_nullable=playtype,
                                                        type_grouping_nullable='offensive',
                                                        player_or_team_abbreviation='P'
                                                        )
            playtype_data = playtype_data.get_data_frames()[0]
            playtype_data_df = pd.DataFrame(playtype_data)

            if playtype_data_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue

            df = playtype_data_df

            df['SEASON'] = season_year

            # Reorder columns to place 'SEASON' and at the start
            df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]


            yield df

def player_defensive_playtype_data_scraper():
    for year in year_iterator_2():
        
        playtype_list = ['Isolation', 'Transition', 'PRBallHandler', 'PRRollMan', 'Postup', 'Spotup', 'Handoff',
                        'Cut', 'OffScreen', 'OffRebound']

        for playtype in playtype_list:
            
            print(f"Fetching {playtype} data for {year}...")
            season_year = year
            season_type = "Regular Season"

            playtype_data = synergyplaytypes.SynergyPlayTypes(season=season_year,
                                                        season_type_all_star=season_type,
                                                        per_mode_simple='PerGame',
                                                        play_type_nullable=playtype,
                                                        type_grouping_nullable='defensive',
                                                        player_or_team_abbreviation='P'
                                                        )
            playtype_data = playtype_data.get_data_frames()[0]
            playtype_data_df = pd.DataFrame(playtype_data)

            if playtype_data_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue

            df = playtype_data_df

            df['SEASON'] = season_year

            # Reorder columns to place 'SEASON' and at the start
            df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]


            yield df

def team_offensive_playtype_data_scraper():
    for year in year_iterator_2():
        
        playtype_list = ['Isolation', 'Transition', 'PRBallHandler', 'PRRollMan', 'Postup', 'Spotup', 'Handoff',
                        'Cut', 'OffScreen', 'OffRebound']

        for playtype in playtype_list:
            
            print(f"Fetching {playtype} data for {year}...")
            season_year = year
            season_type = "Regular Season"

            playtype_data = synergyplaytypes.SynergyPlayTypes(season=season_year,
                                                        season_type_all_star=season_type,
                                                        per_mode_simple='PerGame',
                                                        play_type_nullable=playtype,
                                                        type_grouping_nullable='offensive',
                                                        player_or_team_abbreviation='T'
                                                        )
            playtype_data = playtype_data.get_data_frames()[0]
            playtype_data_df = pd.DataFrame(playtype_data)

            if playtype_data_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue

            df = playtype_data_df

            df['SEASON'] = season_year

            # Reorder columns to place 'SEASON' and at the start
            df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]


            yield df

def team_defensive_playtype_data_scraper():
    for year in year_iterator_2():
        
        playtype_list = ['Isolation', 'Transition', 'PRBallHandler', 'PRRollMan', 'Postup', 'Spotup', 'Handoff',
                        'Cut', 'OffScreen', 'OffRebound']

        for playtype in playtype_list:
            
            print(f"Fetching {playtype} data for {year}...")
            season_year = year
            season_type = "Regular Season"

            playtype_data = synergyplaytypes.SynergyPlayTypes(season=season_year,
                                                        season_type_all_star=season_type,
                                                        per_mode_simple='PerGame',
                                                        play_type_nullable=playtype,
                                                        type_grouping_nullable='defensive',
                                                        player_or_team_abbreviation='T'
                                                        )
            playtype_data = playtype_data.get_data_frames()[0]
            playtype_data_df = pd.DataFrame(playtype_data)

            if playtype_data_df.empty:
                print(f"Skipping {season_year}: No data available.")
                continue

            df = playtype_data_df

            df['SEASON'] = season_year

            # Reorder columns to place 'SEASON' and at the start
            df = df[['SEASON'] + [col for col in df.columns if col != 'SEASON']]


            yield df

#check if functions work properly
# for data in team_defensive_playtype_data_scraper():
#     print(data)


def save_to_csv(data_generator, filename):
    """Saves new season data without duplicates."""
    
    # Load existing CSV data (if it exists)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        existing_seasons = set(existing_df['SEASON'].astype(str))  # Track existing years
    else:
        existing_df = pd.DataFrame()
        existing_seasons = set()

    new_data = []  # List to store new data

    for df in data_generator:
        season_year = str(df['SEASON'].iloc[0])  # Get the season year from the new data
        
        if season_year in existing_seasons:
            print(f"⚠️ Skipping {season_year} (already exists in {filename})")
        else:
            print(f"✅ Adding {season_year} to {filename}")
            new_data.append(df)

    # If there's new data, append and save
    if new_data:
        combined_df = pd.concat([existing_df] + new_data, ignore_index=True)
        combined_df.to_csv(filename, index=False)
        print(f"✅ Data saved to {filename} without duplicates!")
    else:
        print(f"✅ No new data to save for {filename}.")

def csv_generator():
    # save_to_csv(player_base_scraper(), "NBA_PLAYER_BASE.csv")
    # save_to_csv(player_scoring_scraper(), "NBA_PLAYER_SCORING.csv")
    # save_to_csv(team_scoring_scraper(), "NBA_TEAM_SCORING.csv")
    save_to_csv(player_shots_data_scraper(), "NBA_PLAYER_SHOTS.csv")
    # save_to_csv(team_shots_data_scraper(), "NBA_TEAM_SHOTS.csv")
    # save_to_csv(player_offensive_playtype_data_scraper(), "NBA_PLAYER_OFFENSIVE_PLAYTYPE.csv")
    # save_to_csv(team_offensive_playtype_data_scraper(), "NBA_TEAM_OFFENSIVE_PLAYTYPE.csv")
    # save_to_csv(player_defensive_playtype_data_scraper(), "NBA_PLAYER_DEFENSIVE_PLAYTYPE.csv")
    # save_to_csv(team_defensive_playtype_data_scraper(), "NBA_TEAM_DEFENSIVE_PLAYTYPE.csv")
    

#generate the csv's
csv_generator()

# Record the end time
end_time = time.time()

# Calculate the total time taken
elapsed_time = end_time - start_time
print(f"Code execution took {elapsed_time:.2f} seconds.")