import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./augmented_23-24.csv')

# Ensure numeric data for calculations (if columns might have been saved as strings)
df['home_spread'] = pd.to_numeric(df['home_spread'], errors='coerce')
df['odds_spread'] = pd.to_numeric(df['odds_spread'], errors='coerce')

# Calculate the absolute difference between odds_spread and actual spread
df['spread_difference'] = abs(df['odds_spread'] - df['home_spread'])

# Calculate the average difference
average_difference = df['spread_difference'].mean()

print(f"Average difference between odds spread and actual spread: {average_difference:.2f}")

new_df = df[['teams.home.name', 'teams.away.name', 'home_spread', 'odds_spread']]
new_df.to_csv('spreads_and_odds.csv')






# Assuming 'spread_difference' was calculated as abs(df['odds_spread'] - df['home_spread'])
plt.hist(df['spread_difference'], bins=30, edgecolor='k')
plt.title('Distribution of Spread Errors')
plt.xlabel('Error (points)')
plt.ylabel('Frequency')
plt.savefig('experiment.jpg')


median_error = df['spread_difference'].median()
percentiles = df['spread_difference'].quantile([0.25, 0.5, 0.75])

print(f"Median Error: {median_error}")
print(f"25th, 50th (Median), 75th Percentiles:\n{percentiles}")