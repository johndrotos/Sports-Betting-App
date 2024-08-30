import pandas as pd
import json


def initialize_data(dataframe):
    # Dropping Unnecesary Columns
    og_df = dataframe
    df = og_df.drop(['stage', 'status.short', 'status.timer', 'week', 'time', 'timezone', 'status.long',
                    'league.id', 'league.name', 'league.type', 'league.logo', 'country.id', 'country.name', 
                    'country.code', 'country.flag', 'timestamp', 'teams.home.logo', 'teams.away.logo'], axis=1)

    # Ensure data is sorted by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Filling in missing overtime values to 0
    df['scores.home.over_time'].fillna(0, inplace=True)
    df['scores.away.over_time'].fillna(0, inplace=True)

    # Calculating Spreads
    df = df.assign(home_spread=df['scores.home.total']-df['scores.away.total'])

    print(df.shape)
    return df

def average_points(df):
    df['home_avg_points_scored_so_far'] = 0.0
    df['home_avg_points_allowed_so_far'] = 0.0
    df['away_avg_points_scored_so_far'] = 0.0
    df['away_avg_points_allowed_so_far'] = 0.0

    # Create dictionaries to store cumulative stats for each team
    cumulative_stats = {}

    # Loop through each game row to calculate cumulative averages
    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']

        # If team not in dictionary, initialize it
        if home_team not in cumulative_stats:
            cumulative_stats[home_team] = {'points_scored': [], 'points_allowed': []}
        if away_team not in cumulative_stats:
            cumulative_stats[away_team] = {'points_scored': [], 'points_allowed': []}

        # Update cumulative average for home team
        home_scored = cumulative_stats[home_team]['points_scored']
        home_allowed = cumulative_stats[home_team]['points_allowed']
        
        if len(home_scored) > 0:
            df.at[index, 'home_avg_points_scored_so_far'] = sum(home_scored) / len(home_scored)
            df.at[index, 'home_avg_points_allowed_so_far'] = sum(home_allowed) / len(home_allowed)
        
        # Update cumulative average for away team
        away_scored = cumulative_stats[away_team]['points_scored']
        away_allowed = cumulative_stats[away_team]['points_allowed']
        
        if len(away_scored) > 0:
            df.at[index, 'away_avg_points_scored_so_far'] = sum(away_scored) / len(away_scored)
            df.at[index, 'away_avg_points_allowed_so_far'] = sum(away_allowed) / len(away_allowed)
        
        # Update cumulative statistics with current game data
        cumulative_stats[home_team]['points_scored'].append(row['scores.home.total'])
        cumulative_stats[home_team]['points_allowed'].append(row['scores.away.total'])
        
        cumulative_stats[away_team]['points_scored'].append(row['scores.away.total'])
        cumulative_stats[away_team]['points_allowed'].append(row['scores.home.total'])

    return df

def L10_average_points(df):
    df['home_avg_points_scored_L10'] = 0.0
    df['home_avg_points_allowed_L10'] = 0.0
    df['away_avg_points_scored_L10'] = 0.0
    df['away_avg_points_allowed_L10'] = 0.0

    rolling_stats = {}

    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']
        
        # Initialize rolling stats if team not in dictionary
        if home_team not in rolling_stats:
            rolling_stats[home_team] = {'points_scored': [], 'points_allowed': []}
        if away_team not in rolling_stats:
            rolling_stats[away_team] = {'points_scored': [], 'points_allowed': []}
        
        # Calculate rolling average for home team
        home_scored = rolling_stats[home_team]['points_scored']
        home_allowed = rolling_stats[home_team]['points_allowed']
        
        if len(home_scored) > 0:
            df.at[index, 'home_avg_points_scored_L10'] = sum(home_scored) / len(home_scored)
            df.at[index, 'home_avg_points_allowed_L10'] = sum(home_allowed) / len(home_allowed)
        
        # Calculate rolling average for away team
        away_scored = rolling_stats[away_team]['points_scored']
        away_allowed = rolling_stats[away_team]['points_allowed']
        
        if len(away_scored) > 0:
            df.at[index, 'away_avg_points_scored_L10'] = sum(away_scored) / len(away_scored)
            df.at[index, 'away_avg_points_allowed_L10'] = sum(away_allowed) / len(away_allowed)
        
        # Update rolling statistics with current game data
        # Keep only the last 10 games for each team
        rolling_stats[home_team]['points_scored'].append(row['scores.home.total'])
        rolling_stats[home_team]['points_allowed'].append(row['scores.away.total'])
        
        rolling_stats[away_team]['points_scored'].append(row['scores.away.total'])
        rolling_stats[away_team]['points_allowed'].append(row['scores.home.total'])
        
        # Ensure only the last 10 games are kept in the list
        if len(rolling_stats[home_team]['points_scored']) > 10:
            rolling_stats[home_team]['points_scored'].pop(0)
            rolling_stats[home_team]['points_allowed'].pop(0)
        
        if len(rolling_stats[away_team]['points_scored']) > 10:
            rolling_stats[away_team]['points_scored'].pop(0)
            rolling_stats[away_team]['points_allowed'].pop(0)
    
    return df

def main():
    og_df = pd.read_csv('og_games_data.csv')
    df = initialize_data(og_df)
    df = average_points(df)
    df = L10_average_points(df)

    df.insert(20, 'CUMULATIVE_AVERAGES', 'AVERAGES:')
    df.insert(25, 'L10_AVERAGES', 'L10AVERAGES:')

    #Saving the new file
    df.to_csv('./formatted_games_data.csv', index=False)

if __name__ == "__main__":
    main()