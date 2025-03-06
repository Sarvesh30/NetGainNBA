import pandas as pd
import time

start_time = time.time()


def STATS_CLEANED():
    
    def base_cleaner(): #base
        base_df = pd.read_csv("NBA_PLAYER_BASE.csv")
        
        base_df = base_df[["SEASON","PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","AGE","GP","W","L","W_PCT","MIN",
                        "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","AST","OREB","DREB","REB",
                        "TOV","STL","BLK","BLKA","PF","PFD","PLUS_MINUS"]]
        
        return base_df

    def adv_cleaner(): #advanced
        adv_df = pd.read_csv("NBA_PLAYER_ADVANCED.csv")
        adv_df = adv_df[["SEASON","PLAYER_ID","TEAM_ID","OFF_RATING","DEF_RATING",
                        "NET_RATING","AST_PCT","AST_TO","AST_RATIO","OREB_PCT","DREB_PCT","REB_PCT",
                        "TM_TOV_PCT","E_TOV_PCT","EFG_PCT","TS_PCT","USG_PCT","PACE","PIE","POSS"]]

        return adv_df
    
    def def_cleaner(): #defense
        def_df = pd.read_csv("NBA_PLAYER_DEFENSE.csv")
        def_df = def_df[["SEASON","PLAYER_ID","TEAM_ID",
                        "OPP_PTS_OFF_TOV","OPP_PTS_2ND_CHANCE","OPP_PTS_FB","OPP_PTS_PAINT","DEF_WS"]]
        
        return def_df
    
    def usg_cleaner(): #usage
        usg_df = pd.read_csv("NBA_PLAYER_USAGE.csv")
        usg_df = usg_df[["SEASON","PLAYER_ID","TEAM_ID","PCT_PTS","PCT_FGM","PCT_FGA","PCT_FG3M",
                        "PCT_FG3A","PCT_FTM","PCT_FTA","PCT_OREB","PCT_DREB","PCT_REB","PCT_AST","PCT_TOV","PCT_STL","PCT_BLK",
                        "PCT_BLKA","PCT_PF","PCT_PFD"]]
        
        return usg_df
    
    def misc_cleaner(): #miscellaneous
        misc_df = pd.read_csv("NBA_PLAYER_MISC.csv")
        misc_df = misc_df[["SEASON","PLAYER_ID","TEAM_ID","PTS_OFF_TOV","PTS_2ND_CHANCE","PTS_FB","PTS_PAINT"]]
        
        return misc_df

    def scor_cleaner(): #scoring data
        scor_df = pd.read_csv("NBA_PLAYER_SCORING.csv")
        scor_df = scor_df[["SEASON","PLAYER_ID","TEAM_ID","PCT_FGA_2PT","PCT_FGA_3PT","PCT_PTS_2PT","PCT_PTS_2PT_MR",
                        "PCT_PTS_3PT","PCT_PTS_FB","PCT_PTS_FT","PCT_PTS_OFF_TOV","PCT_PTS_PAINT","PCT_AST_2PM","PCT_UAST_2PM",
                        "PCT_AST_3PM","PCT_UAST_3PM","PCT_AST_FGM","PCT_UAST_FGM"]]
        
        return scor_df

    def shots_cleaner(): #shots data
        shots_df = pd.read_csv("NBA_PLAYER_SHOTS.csv")
        shots_df = shots_df[["SEASON","PLAYER_ID","TEAM_ID","RESTRICTED_AREA_FGM","RESTRICTED_AREA_FGA",
                            "RESTRICTED_AREA_FG_PCT","IN_THE_PAINT_(NON-RA)_FGM","IN_THE_PAINT_(NON-RA)_FGA",
                            "IN_THE_PAINT_(NON-RA)_FG_PCT","MID-RANGE_FGM","MID-RANGE_FGA","MID-RANGE_FG_PCT",
                            "CORNER_3_FGM","CORNER_3_FGA","CORNER_3_FG_PCT","ABOVE_THE_BREAK_3_FGM",
                            "ABOVE_THE_BREAK_3_FGA","ABOVE_THE_BREAK_3_FG_PCT"]]
        
        return shots_df

    # Load & Clean Datasets
    base_df = base_cleaner()
    adv_df = adv_cleaner()
    def_df = def_cleaner()
    usg_df = usg_cleaner()
    misc_df = misc_cleaner()
    scor_df = scor_cleaner()
    shots_df = shots_cleaner()

    # Merge All Cleaned DataFrames based on ["SEASON", "PLAYER_ID", "TEAM_ID"]
    merged_df = base_df \
        .merge(adv_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \
        .merge(def_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \
        .merge(usg_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \
        .merge(misc_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \
        .merge(scor_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \
        .merge(shots_df, on=["SEASON", "PLAYER_ID", "TEAM_ID"], how="left") \

    # Save Final Cleaned & Merged Dataset
    merged_df.to_csv("NBA_PLAYER_FINAL_DATASET.csv", index=False)

    print("Merged dataset successfully created with shape:", merged_df.shape)

STATS_CLEANED()


end_time = time.time()

#calculate time code takes to run
elapsed_time = end_time - start_time
print(f"The datasets have been cleaned! Total process time: {elapsed_time:.2f}")