import pandas as pd

df = pd.read_csv('./all_historical_odds.csv')

df = df.drop_duplicates(subset=['Game ID', 'Point'])

df.to_csv('all_historical_odds.csv', index=False)