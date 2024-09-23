import requests
import pandas as pd
import json

### PARAM INFO ###
# NBA is league 12
# The API contains info back to the 2008-2009 season


API_KEY = 'c4d5236bd0mshbc6576c04564c42p1e9e60jsned8d838ec790'  # Replace with your actual API key
url = "https://api-basketball.p.rapidapi.com/games"

payload={}
headers = {
'x-rapidapi-key': API_KEY,
'x-rapidapi-host': 'api-basketball.p.rapidapi.com'
}

params = {
    "league":"12",
    "season":"2023-2024"
}

response = requests.get(url, headers=headers,params=params)



if response.status_code == 200:
    data = response.json()['response']
    df = pd.json_normalize(data)
    df.to_csv('./raw23-24.csv', index=False)
    print(df.shape)
else:
    print(f"Error: {response.status_code}")