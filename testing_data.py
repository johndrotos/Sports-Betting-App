import pandas as pd

def main():
    df = pd.read_csv('./formatted_data/23-24.csv')
    knicks_games = df[(df['teams.away.id'] == 151) | (df['teams.home.id'] == 151)]
    knicks_games = knicks_games['date', 'teams.home.name', 'teams.away.name', 'scores.home.total', 'scores.away.total']
    knicks_games.to_csv('./knicks23-24.csv')
    
if __name__ == "__main__":
    main()