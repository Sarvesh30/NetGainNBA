import pandas as pd

def STATS_CLEANED():
    
    def base_cleaner(): # Base stats
        base_df = pd.read_csv("NBA_TEAM_BASE.csv")
        base_df = base_df[["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", 
                           "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", 
                           "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS", "GP_RANK", 
                           "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", 
                           "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK", "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", 
                           "OREB_RANK", "DREB_RANK", "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", 
                           "BLKA_RANK", "PF_RANK", "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK"]]
        return base_df
    
    def adv_cleaner(): # Advanced stats
        adv_df = pd.read_csv("NBA_TEAM_ADVANCED.csv")
        adv_df = adv_df[["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", 
                         "PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_2PT_MR", "PCT_PTS_3PT", 
                         "PCT_PTS_FB", "PCT_PTS_FT", "PCT_PTS_OFF_TOV", "PCT_PTS_PAINT", "PCT_AST_2PM", 
                         "PCT_UAST_2PM", "PCT_AST_3PM", "PCT_UAST_3PM", "PCT_AST_FGM", "PCT_UAST_FGM"]]
        return adv_df
    
    def def_cleaner(): # Defensive stats
        def_df = pd.read_csv("NBA_TEAM_DEFENSE.csv")
        def_df = def_df[["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", "DEF_RATING", 
                         "DREB", "DREB_PCT", "STL", "BLK", "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE", 
                         "OPP_PTS_FB", "OPP_PTS_PAINT"]]
        return def_df
    
    def misc_cleaner(): # Miscellaneous stats
        misc_df = pd.read_csv("NBA_TEAM_MISC.csv")
        misc_df = misc_df[["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", "PTS_OFF_TOV", 
                           "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT", "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE", 
                           "OPP_PTS_FB", "OPP_PTS_PAINT"]]
        return misc_df
    
    def shots_cleaner(): # Shots data
        shots_df = pd.read_csv("NBA_TEAM_SHOTS.csv")
        shots_df = shots_df[["SEASON", "TEAM_ID", "TEAM_NAME", "Restricted Area FGM", "Restricted Area FGA", 
                             "Restricted Area FG_PCT", "In The Paint (Non-RA) FGM", "In The Paint (Non-RA) FGA", 
                             "In The Paint (Non-RA) FG_PCT", "Mid-Range FGM", "Mid-Range FGA", "Mid-Range FG_PCT", 
                             "Left Corner 3 FGM", "Left Corner 3 FGA", "Left Corner 3 FG_PCT", "Right Corner 3 FGM", 
                             "Right Corner 3 FGA", "Right Corner 3 FG_PCT", "Above the Break 3 FGM", "Above the Break 3 FGA", 
                             "Above the Break 3 FG_PCT", "Backcourt FGM", "Backcourt FGA", "Backcourt FG_PCT", 
                             "Corner 3 FGM", "Corner 3 FGA", "Corner 3 FG_PCT"]]
        return shots_df
    
    def scor_cleaner(): # Scoring data
        scor_df = pd.read_csv("NBA_TEAM_SCORING.csv")
        scor_df = scor_df[["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "MIN", "PCT_FGA_2PT", 
                           "PCT_FGA_3PT", "PCT_PTS_2PT", "PCT_PTS_2PT_MR", "PCT_PTS_3PT", "PCT_PTS_FB", "PCT_PTS_FT", 
                           "PCT_PTS_OFF_TOV", "PCT_PTS_PAINT", "PCT_AST_2PM", "PCT_UAST_2PM", "PCT_AST_3PM", 
                           "PCT_UAST_3PM", "PCT_AST_FGM", "PCT_UAST_FGM"]]
        return scor_df
    
    # Load & Clean Datasets
    base_df = base_cleaner()
    adv_df = adv_cleaner()
    def_df = def_cleaner()
    misc_df = misc_cleaner()
    scor_df = scor_cleaner()
    shots_df = shots_cleaner()
    
    # Merge All Cleaned DataFrames
    merged_df = base_df.merge(adv_df, on=["SEASON", "TEAM_ID"], how="left", suffixes=("", "_adv")) \
                       .merge(def_df, on=["SEASON", "TEAM_ID"], how="left", suffixes=("", "_def")) \
                       .merge(misc_df, on=["SEASON", "TEAM_ID"], how="left", suffixes=("", "_misc")) \
                       .merge(scor_df, on=["SEASON", "TEAM_ID"], how="left", suffixes=("", "_scor")) \
                       .merge(shots_df, on=["SEASON", "TEAM_ID"], how="left", suffixes=("", "_shots"))
    
    merged_df.to_csv("NBA_TEAM_FINAL_DATASET.csv", index=False)

STATS_CLEANED()