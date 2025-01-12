import pandas as pd
import numpy as np
import json


def initialize_data(dataframe):
    # Dropping Unnecesary Columns
    og_df = dataframe
    df = og_df.drop(['stage', 'status.short', 'venue', 'status.timer', 'week', 'time', 'timezone', 'status.long',
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
    df['winner'] = np.where(df['home_spread'] > 0, 1, 0)
    df = df.assign(total=df['scores.home.total']+df['scores.away.total'])

    

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

        # Include streaks
        df.at[index, 'home_streak'] = records[home_team]['streak']
        df.at[index, 'away_streak'] = records[away_team]['streak']

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

def remove_pre_post(df, start_date, end_date):
    df = df[df['date'] >= start_date]
    df = df[df['date'] < end_date]
    return df

def remove_first_10(df):
    df['game_number'] = df.groupby(['teams.home.name']).cumcount() + 1
    df = df[df['game_number'] > 10]

    return df.drop(columns=['game_number'])  # Drop 'game_number' after filtering




def format(raw_data_loc, output_loc, season_start_date, season_end_date):
    og_df = pd.read_csv(raw_data_loc)
    save_location = output_loc
    df = initialize_data(og_df)
    df = remove_pre_post(df, season_start_date, season_end_date)

    # Feature engineering
    df = win_percentages(df)
    df = average_points(df)
    df = L10_average_points(df)
    df = L5_average_points_home_and_road(df)
    df = calculate_days_since_last_game(df)
    df = head2head(df)

    # Round Columns to 2 decimal points
    df = df.round({'home_win_percentage': 1,
                   'away_win_percentage': 1,
                   'home_avg_points_scored_combined': 1,
                   'home_avg_points_allowed_combined': 1,
                   'away_avg_points_scored_combined': 1,
                   'away_avg_points_allowed_combined': 1,
                   'home_avg_points_scored_at_home': 1,
                   'home_avg_points_allowed_at_home': 1,
                   'away_avg_points_scored_on_road': 1,
                   'away_avg_points_allowed_on_road': 1
                   })


    i = 22
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

    # Remove first 10 games
    df = remove_first_10(df)
    
    # Remove quarterly scores
    columns_to_drop = ['scores.home.quarter_1', 'scores.home.quarter_2', 'scores.home.quarter_3', 
                       'scores.home.quarter_4', 'scores.home.over_time', 'scores.away.quarter_1', 'scores.away.quarter_2', 
                       'scores.away.quarter_3', 'scores.away.quarter_4', 'scores.away.over_time']
    df = df.drop(columns=columns_to_drop)
    
    # Saving the new file
    df.to_csv(save_location, index=False)


def main():
    
    #       input                       output                      start           end
    format("./raw_data/raw08-09.csv", './formatted_data/08-09.csv', '2008-10-28', '2009-04-15')
    format("./raw_data/raw09-10.csv", './formatted_data/09-10.csv', '2009-10-27', '2010-04-14')
    format("./raw_data/raw10-11.csv", './formatted_data/10-11.csv', '2010-10-26', '2011-04-13')
    format("./raw_data/raw11-12.csv", './formatted_data/11-12.csv', '2011-12-25', '2012-04-26')
    format("./raw_data/raw12-13.csv", './formatted_data/12-13.csv', '2012-10-30', '2013-04-17')
    format("./raw_data/raw13-14.csv", './formatted_data/13-14.csv', '2013-10-29', '2014-04-16')
    format("./raw_data/raw14-15.csv", './formatted_data/14-15.csv', '2014-10-28', '2015-04-15')
    format("./raw_data/raw15-16.csv", './formatted_data/15-16.csv', '2015-10-27', '2016-04-13')
    format("./raw_data/raw16-17.csv", './formatted_data/16-17.csv', '2016-10-25', '2017-04-12')
    format("./raw_data/raw17-18.csv", './formatted_data/17-18.csv', '2017-10-17', '2018-04-11')
    format("./raw_data/raw18-19.csv", './formatted_data/18-19.csv', '2018-10-16', '2019-04-10')
    format("./raw_data/raw19-20.csv", './formatted_data/19-20.csv', '2019-10-22', '2020-03-11')  # COVID interruption
    format("./raw_data/raw20-21.csv", './formatted_data/20-21.csv', '2020-12-22', '2021-05-16')  # Shortened season
    format("./raw_data/raw21-22.csv", './formatted_data/21-22.csv', '2021-10-19', '2022-04-10')
    format("./raw_data/raw22-23.csv", './formatted_data/22-23.csv', '2022-10-18', '2023-04-09')
    format("./raw_data/raw23-24.csv", './formatted_data/23-24.csv', '2023-10-24', '2024-04-14')  # Expected end date
    

    
    # Load your individual season DataFrames (these should already be in memory if you used format())
    df_08_09 = pd.read_csv('./formatted_data/08-09.csv')
    df_09_10 = pd.read_csv('./formatted_data/09-10.csv')
    df_10_11 = pd.read_csv('./formatted_data/10-11.csv')
    df_11_12 = pd.read_csv('./formatted_data/11-12.csv')
    df_12_13 = pd.read_csv('./formatted_data/12-13.csv')
    df_13_14 = pd.read_csv('./formatted_data/13-14.csv')
    df_14_15 = pd.read_csv('./formatted_data/14-15.csv')
    df_15_16 = pd.read_csv('./formatted_data/15-16.csv')
    df_16_17 = pd.read_csv('./formatted_data/16-17.csv')
    df_17_18 = pd.read_csv('./formatted_data/17-18.csv')
    df_18_19 = pd.read_csv('./formatted_data/18-19.csv')
    df_19_20 = pd.read_csv('./formatted_data/19-20.csv')
    df_20_21 = pd.read_csv('./formatted_data/20-21.csv')
    df_21_22 = pd.read_csv('./formatted_data/21-22.csv')
    df_22_23 = pd.read_csv('./formatted_data/22-23.csv')
    df_23_24 = pd.read_csv('./formatted_data/23-24.csv')

    # Combine all the DataFrames into one
    all_seasons_df = pd.concat([df_08_09, df_09_10, df_10_11, df_11_12, df_12_13, df_13_14, df_14_15,
                                df_15_16, df_16_17, df_17_18, df_18_19, df_19_20, df_20_21, df_21_22,
                                df_22_23, df_23_24], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    all_seasons_df.to_csv('./formatted_data/all_seasons.csv', index=False)
    

    df = pd.read_csv('./formatted_data/all_seasons.csv')

    columns_to_drop = ['date', 'teams.home.name', 'teams.away.name', 
                        'scores.home.total', 'scores.away.total', 'WIN_PERCENTAGES', 'CUMULATIVE_AVERAGES', 'AT_HOME_AVGS', 'ON_ROAD_AVGS', 'L10_AVERAGES', 
                       'L5_AVERAGES', 'DAYS_SINCE', 'H2H']
    df = df.drop(columns=columns_to_drop)


    print(df.isnull().sum())

    rows_with_nulls = df[df.isnull().any(axis=1)]

    print(rows_with_nulls)

    df.to_csv('./formatted_data/training_data.csv', index=False)



print("Successfully merged all season dataframes!")



if __name__ == "__main__":
    main()