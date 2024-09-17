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
    df['home_avg_points_scored_combined'] = 0.0
    df['home_avg_points_allowed_combined'] = 0.0
    df['away_avg_points_scored_combined'] = 0.0
    df['away_avg_points_allowed_combined'] = 0.0
    df['home_avg_points_scored_at_home'] = 0.0
    df['home_avg_points_allowed_at_home'] = 0.0
    df['away_avg_points_scored_on_road'] = 0.0
    df['away_avg_points_allowed_on_road'] = 0.0

    cumulative_stats = {}

    # Function to initialize the home and away stats for a team
    def initialize_team(team):
        if team not in cumulative_stats:
            cumulative_stats[team] = {
                'points_scored_combined': [],  # Overall points scored (home + away)
                'points_allowed_combined': [],  # Overall points allowed (home + away)
                'points_scored_at_home': [],  # Points scored at home
                'points_allowed_at_home': [],  # Points allowed at home
                'points_scored_on_road': [],  # Points scored away
                'points_allowed_on_road': []  # Points allowed away
            }

    # Loop through each game row to calculate cumulative averages
    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']

        # Initialize both home and away teams (ensures both teams are fully initialized)
        initialize_team(home_team)
        initialize_team(away_team)

        # Cumulative averages for both teams
        home_scored = cumulative_stats[home_team]['points_scored_combined']
        home_allowed = cumulative_stats[home_team]['points_allowed_combined']
        away_scored = cumulative_stats[away_team]['points_scored_combined']
        away_allowed = cumulative_stats[away_team]['points_allowed_combined']

        # Update cumulative averages for the home team
        if len(home_scored) > 0:
            df.at[index, 'home_avg_points_scored_combined'] = sum(home_scored) / len(home_scored)
            df.at[index, 'home_avg_points_allowed_combined'] = sum(home_allowed) / len(home_allowed)

        # Update cumulative averages for the away team
        if len(away_scored) > 0:
            df.at[index, 'away_avg_points_scored_combined'] = sum(away_scored) / len(away_scored)
            df.at[index, 'away_avg_points_allowed_combined'] = sum(away_allowed) / len(away_allowed)

        # Cumulative home-specific and away-specific averages
        home_scored_home = cumulative_stats[home_team]['points_scored_at_home']
        home_allowed_home = cumulative_stats[home_team]['points_allowed_at_home']
        away_scored_away = cumulative_stats[away_team]['points_scored_on_road']
        away_allowed_away = cumulative_stats[away_team]['points_allowed_on_road']

        # Home averages (only home games)
        if len(home_scored_home) > 0:
            df.at[index, 'home_avg_points_scored_at_home'] = sum(home_scored_home) / len(home_scored_home)
            df.at[index, 'home_avg_points_allowed_at_home'] = sum(home_allowed_home) / len(home_allowed_home)

        # Away averages (only away games)
        if len(away_scored_away) > 0:
            df.at[index, 'away_avg_points_scored_on_road'] = sum(away_scored_away) / len(away_scored_away)
            df.at[index, 'away_avg_points_allowed_on_road'] = sum(away_allowed_away) / len(away_allowed_away)

        # Update cumulative statistics with current game data
        # Home team stats (both overall and home-specific)
        cumulative_stats[home_team]['points_scored_combined'].append(row['scores.home.total'])
        cumulative_stats[home_team]['points_allowed_combined'].append(row['scores.away.total'])
        cumulative_stats[home_team]['points_scored_at_home'].append(row['scores.home.total'])
        cumulative_stats[home_team]['points_allowed_at_home'].append(row['scores.away.total'])

        # Away team stats (both overall and away-specific)
        cumulative_stats[away_team]['points_scored_combined'].append(row['scores.away.total'])
        cumulative_stats[away_team]['points_allowed_combined'].append(row['scores.home.total'])
        cumulative_stats[away_team]['points_scored_on_road'].append(row['scores.away.total'])
        cumulative_stats[away_team]['points_allowed_on_road'].append(row['scores.home.total'])

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

def L5_average_points_home_and_road(df):
    df['home_avg_points_scored_L5_at_home'] = 0.0
    df['home_avg_points_allowed_L5_at_home'] = 0.0
    df['away_avg_points_scored_L5_on_road'] = 0.0
    df['away_avg_points_allowed_L5_on_road'] = 0.0

    rolling_stats = {}

    def initialize_team(team):
        if team not in rolling_stats:
            rolling_stats[team] = {
                'points_scored_home': [],
                'points_allowed_home': [],
                'points_scored_away': [],
                'points_allowed_away': []
            }

    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']
        
        initialize_team(home_team)
        initialize_team(away_team)
        
        # Calculate rolling average for home team at home
        home_scored_home = rolling_stats[home_team]['points_scored_home']
        home_allowed_home = rolling_stats[home_team]['points_allowed_home']
        if len(home_scored_home) > 0:
            df.at[index, 'home_avg_points_scored_L5_at_home'] = sum(home_scored_home) / len(home_scored_home)
            df.at[index, 'home_avg_points_allowed_L5_at_home'] = sum(home_allowed_home) / len(home_allowed_home)
        
        # Calculate rolling average for away team on the road
        away_scored_away = rolling_stats[away_team]['points_scored_away']
        away_allowed_away = rolling_stats[away_team]['points_allowed_away']
        if len(away_scored_away) > 0:
            df.at[index, 'away_avg_points_scored_L5_on_road'] = sum(away_scored_away) / len(away_scored_away)
            df.at[index, 'away_avg_points_allowed_L5_on_road'] = sum(away_allowed_away) / len(away_allowed_away)
        
        # Update rolling statistics with current game data
        rolling_stats[home_team]['points_scored_home'].append(row['scores.home.total'])
        rolling_stats[home_team]['points_allowed_home'].append(row['scores.away.total'])

        rolling_stats[away_team]['points_scored_away'].append(row['scores.away.total'])
        rolling_stats[away_team]['points_allowed_away'].append(row['scores.home.total'])

        # Ensure only the last 5 games are kept in the list
        if len(rolling_stats[home_team]['points_scored_home']) > 5:
            rolling_stats[home_team]['points_scored_home'].pop(0)
            rolling_stats[home_team]['points_allowed_home'].pop(0)
        
        if len(rolling_stats[away_team]['points_scored_away']) > 5:
            rolling_stats[away_team]['points_scored_away'].pop(0)
            rolling_stats[away_team]['points_allowed_away'].pop(0)
    
    return df

def calculate_days_since_last_game(df):
    # Initialize columns to store the days since the last game for home and away teams
    df['home_days_since_last_game'] = None
    df['away_days_since_last_game'] = None

    # Create a dictionary to track the last game date for each team
    last_game_date = {}

    # Loop through each game row to calculate days since last game
    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']
        game_date = pd.to_datetime(row['date'])  # Ensure the date is in datetime format

        # Calculate days since last game for home team
        if home_team in last_game_date:
            days_since_home = (game_date - last_game_date[home_team]).days
            df.at[index, 'home_days_since_last_game'] = days_since_home
        else:
            df.at[index, 'home_days_since_last_game'] = 0  # Or set to a default value, e.g., 0
        last_game_date[home_team] = game_date

        # Calculate days since last game for away team
        if away_team in last_game_date:
            days_since_away = (game_date - last_game_date[away_team]).days
            df.at[index, 'away_days_since_last_game'] = days_since_away
        else:
            df.at[index, 'away_days_since_last_game'] = 0     # Or set to a default value, e.g., 0
        last_game_date[away_team] = game_date

    return df

def win_percentages(df):
    # Create columns
    df['home_win_percentage'] = 0.0
    df['away_win_percentage'] = 0.0
    df['home_streak'] = 0
    df['away_streak'] = 0

    records = {}

    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']

        # Add teams to records dictionary if not present
        for team in home_team, away_team:
            if team not in records:
                records[team] = {'record': [], 'streak': 0}

        # Calculate win % up to now
        home_results = records[home_team]['record']
        if len(home_results) > 0:
            df.at[index, 'home_win_percentage'] = sum(home_results) / len(home_results)
        
        away_results = records[away_team]['record']
        if len(away_results) > 0:
            df.at[index, 'away_win_percentage'] = sum(away_results) / len(away_results)

        # Add current game to records
        if row['home_spread'] > 0: 
            winner = home_team
            loser = away_team
        else:
            winner = away_team
            loser = home_team

        records[winner]['record'].append(1)
        records[loser]['record'].append(0)

        # Update streaks
        records[winner]['streak'] = max(0, records[winner]['streak']) + 1
        records[loser]['streak'] = min(0, records[loser]['streak']) - 1
        df.at[index, 'home_streak'] = records[home_team]['streak']
        df.at[index, 'away_streak'] = records[away_team]['streak']

    return df
        
def head2head(df):
    # Create columns to store H2H metrics
    df['home_h2h_spread'] = 0  # Home spread (home points - away points)
    df['home_h2h_record'] = 0  # Home team's wins vs the away team
    df['away_h2h_record'] = 0  # Away team's wins vs the home team

    # Dictionary to store H2H stats: { ('team1', 'team2'): {'points_scored_team1_home': [], 'points_scored_team2_home': [], 'team1_wins': 0, 'team2_wins': 0}}
    head2head_stats = {}

    # Iterate through each row to calculate H2H metrics
    for index, row in df.iterrows():
        home_team = row['teams.home.name']
        away_team = row['teams.away.name']

        # Create a unified matchup key (sorted alphabetically)
        matchup_key = tuple(sorted([home_team, away_team]))

        # Initialize H2H data for this matchup if not already present
        if matchup_key not in head2head_stats:
            head2head_stats[matchup_key] = {
                'points_scored_team1_home': [],  # Points scored by team1 when they were home
                'points_scored_team2_home': [],  # Points scored by team2 when they were home
                'team1_wins': 0,  # Number of wins by team1 in this matchup
                'team2_wins': 0   # Number of wins by team2 in this matchup
            }

        # Get the current H2H stats
        h2h_stats = head2head_stats[matchup_key]

        # Identify team1 and team2 based on alphabetical order
        team1, team2 = matchup_key

        # Determine if team1 is home or away for this game
        if home_team == team1:
            # team1 is the home team, team2 is the away team for this game
            total_home_points = sum(h2h_stats['points_scored_team1_home'])
            total_away_points = sum(h2h_stats['points_scored_team2_home'])
            df.at[index, 'home_h2h_spread'] = total_home_points - total_away_points
            df.at[index, 'home_h2h_record'] = h2h_stats['team1_wins']
            df.at[index, 'away_h2h_record'] = h2h_stats['team2_wins']
        else:
            # team2 is the home team, team1 is the away team for this game
            total_home_points = sum(h2h_stats['points_scored_team2_home'])
            total_away_points = sum(h2h_stats['points_scored_team1_home'])
            df.at[index, 'home_h2h_spread'] = total_home_points - total_away_points
            df.at[index, 'home_h2h_record'] = h2h_stats['team2_wins']
            df.at[index, 'away_h2h_record'] = h2h_stats['team1_wins']

        # Determine winner and loser for the current game
        if row['home_spread'] > 0:  # Home team wins
            if home_team == team1:
                h2h_stats['team1_wins'] += 1
            else:
                h2h_stats['team2_wins'] += 1
        else:  # Away team wins
            if away_team == team1:
                h2h_stats['team1_wins'] += 1
            else:
                h2h_stats['team2_wins'] += 1

        # Update H2H points with the current game’s results
        if home_team == team1:
            h2h_stats['points_scored_team1_home'].append(row['scores.home.total'])
            h2h_stats['points_scored_team2_home'].append(row['scores.away.total'])
        else:
            h2h_stats['points_scored_team2_home'].append(row['scores.home.total'])
            h2h_stats['points_scored_team1_home'].append(row['scores.away.total'])

    return df

def remove_preseason(df):
    df['date'] = pd.to_datetime(df['date'])
    start_date = pd.to_datetime(2023-10-24)
    








def main():
    og_df = pd.read_csv('raw23-24.csv')
    df = initialize_data(og_df)
    df = win_percentages(df)
    df = average_points(df)
    df = L10_average_points(df)
    df = L5_average_points_home_and_road(df)
    df = calculate_days_since_last_game(df)
    df = head2head(df)

    i = 20
    df.insert(i, 'WIN_PERCENTAGES', 'WIN_PERCENTAGES:')
    i += 5
    df.insert(i, 'CUMULATIVE_AVERAGES', 'CUM_AVERAGES:')
    i += 5
    df.insert(i, 'AT_HOME_AVGS', 'AT_HOME_AVGS:')
    i += 3
    df.insert(i, 'ON_ROAD_AVGS', 'ON_ROAD_AVGS:')
    i += 3
    df.insert(i, 'L10_AVERAGES', 'L10AVERAGES:')
    i += 5
    df.insert(i, 'L5_AVERAGES', 'L5AVERAGES:')
    i += 5
    df.insert(i, 'DAYS_SINCE', "DAYS_SINCE:")
    i += 3
    df.insert(i, 'H2H', 'H2H:')

    #Saving the new file
    df.to_csv('./23-24.csv', index=False)

if __name__ == "__main__":
    main()