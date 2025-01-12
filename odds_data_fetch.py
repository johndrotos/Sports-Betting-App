import requests
import pandas as pd
import json
from datetime import datetime, timedelta



API_KEY = '65ffba3835e1b2356bf206e108225b50'
SPORT = 'basketball_nba'
REGIONS = 'us' 
MARKETS = 'spreads'
BOOKMAKERS = 'fanduel'
ODDS_FORMAT = 'decimal' 
DATE_FORMAT = 'iso'

# DATE = '2024-01-01T12:00:00Z'


start_date = datetime(2023, 11, 25)  
end_date = datetime(2024, 4, 15) 
date_increment = timedelta(hours=24)  

all_data = []

current_date = start_date
while current_date < end_date:
    date_str = current_date.isoformat() + "Z"  # Convert to ISO 8601 format with 'Z' for UTC
    print(f"Fetching data for: {date_str}")

    # API Request
    odds_response = requests.get(
        f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
            'date': date_str,
            'bookmakers': BOOKMAKERS,
        }
    )
    
    if odds_response.status_code == 200:
        odds_json = odds_response.json()
        # Extract relevant data
        for event in odds_json.get('data', []):
            game_id = event['id']
            home_team = event['home_team']
            away_team = event['away_team']
            commence_time = event['commence_time']

            for bookmaker in event.get('bookmakers', []):
                if bookmaker['key'] == BOOKMAKERS:  # Ensure FanDuel odds
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':  # Only process spread markets
                            for outcome in market.get('outcomes', []):
                                all_data.append({
                                    'Game ID': game_id,
                                    'Home Team': home_team,
                                    'Away Team': away_team,
                                    'Commence Time': commence_time,
                                    'Bookmaker': bookmaker['title'],
                                    'Outcome Name': outcome['name'],
                                    'Point': outcome['point'],
                                    'Price': outcome['price'],
                                    'Timestamp': odds_json.get('timestamp'),
                                })

    else:
        print(f"Failed to fetch data for {date_str}: {odds_response.status_code}, {odds_response.text}")

    # Increment date
    current_date += date_increment
    
    
df = pd.DataFrame(all_data)
df = df.drop_duplicates(subset=['Game ID', 'Outcome Name'])
df = df[df['Outcome Name'] == df['Home Team']]

df.to_csv('all_historical_odds.csv', index=False)
print("All historical odds data saved to all_historical_odds.csv")

# Check the usage quota
print('Remaining requests', odds_response.headers['x-requests-remaining'])
print('Used requests', odds_response.headers['x-requests-used'])