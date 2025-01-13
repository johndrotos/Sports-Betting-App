import pandas as pd
from datetime import datetime


odds_df = pd.read_csv('./all_historical_odds.csv')
stats_df = pd.read_csv('formatted_data/23-24.csv')

odds_df['Commence Time'] = pd.to_datetime(odds_df['Commence Time'])
stats_df['date'] = pd.to_datetime(stats_df['date'])



def find_matching_game(row, odds_df):
    filtered_odds = odds_df[
        (odds_df['Home Team'] == row['teams.home.name']) &
        (odds_df['Away Team'] == row['teams.away.name'])
    ]
    if not filtered_odds.empty:
        # Find the closest date match
        filtered_odds = filtered_odds.copy()  # Avoid SettingWithCopyWarning
        filtered_odds['date_diff'] = abs(filtered_odds['Commence Time'] - row['date'])
        closest_match = filtered_odds.loc[filtered_odds['date_diff'].idxmin()]
        return closest_match[['Point']]
    return pd.Series()  # Return an empty series if no match found

# Apply the matching logic
stats_df[['odds_spread']] = stats_df.apply(
    lambda row: find_matching_game(row, odds_df),
    axis=1
)
stats_df['odds_spread'] = -stats_df['odds_spread']

stats_df.to_csv('./augmented_23-24.csv', index=False)
print("Augmented dataset saved to './augmented_23-24.csv'")
