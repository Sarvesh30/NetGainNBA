import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres.hogctscrviurutvkjvzf:Aim25NetGainNBA@aws-0-us-east-2.pooler.supabase.com:5432/postgres"

engine = create_engine(DATABASE_URL)

CSV_LIST = ["NBA_PLAYER_SCORING.csv", "NBA_TEAM_SCORING.csv", "NBA_PLAYER_SHOTS.csv", "NBA_TEAM_SHOTS.csv", 
            "NBA_PLAYER_OFFENSIVE_PLAYTYPE.csv", "NBA_TEAM_OFFENSIVE_PLAYTYPE.csv", "NBA_PLAYER_DEFENSIVE_PLAYTYPE.csv", 
            "NBA_TEAM_DEFENSIVE_PLAYTYPE.csv"]

for FILE_NAME in CSV_LIST:

    df = pd.read_csv(FILE_NAME)

    try:
        df.to_sql(FILE_NAME.replace(".csv",""), engine, if_exists="replace", index=False)
        print(f"{FILE_NAME} data uploaded successfully!")
    except Exception as e:
        print(f"Error uploading data: {e}")