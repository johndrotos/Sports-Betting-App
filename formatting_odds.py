import pandas as pd

df = pd.read_csv('./all_historical_odds.csv')
print(len(df))

df = df.drop_duplicates(subset=['Game ID'])
print(len(df))

df.to_csv('all_historical_odds.csv', index=False)