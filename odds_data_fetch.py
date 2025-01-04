import requests
import pandas as pd
import json

# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
API_KEY = '65ffba3835e1b2356bf206e108225b50'

SPORT = 'basketball_nba' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports

REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'spreads' # h2h | spreads | totals. Multiple can be specified if comma delimited

BOOKMAKERS = 'draftkings, fanduel, betmgm'

ODDS_FORMAT = 'decimal' # decimal | american

DATE_FORMAT = 'iso'

DATE = '2021-01-23T12:00:00Z'


odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        #'bookmakers': BOOKMAKERS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
        'date': DATE
    }
)

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    print(odds_json)

    with open("data.json", "w") as f:
        json.dump(odds_json, f, indent=4)

    data = []
    for event in odds_json.get('data', []):
        game_id = event['id']
        home_team = event['home_team']
        away_team = event['away_team']
        commence_time = event['commence_time']
        
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'spreads':  # We're interested in spread odds
                    for outcome in market.get('outcomes', []):
                        data.append({
                            'Game ID': game_id,
                            'Home Team': home_team,
                            'Away Team': away_team,
                            'Commence Time': commence_time,
                            'Bookmaker': bookmaker['title'],
                            'Outcome Name': outcome['name'],
                            'Point': outcome['point'],
                            'Price': outcome['price'],
                        })

    df = pd.DataFrame(data)
    print(df)
    df.to_csv('fanduel_nba_odds.csv', index=False)
    print("Data saved to fanduel_nba_odds.csv")



    # Check the usage quota
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])