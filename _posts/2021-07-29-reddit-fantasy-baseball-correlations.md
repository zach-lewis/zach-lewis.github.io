# Analysis of Player Mentions on r/fantasybaseball to date
The fantasybaseball_praw program (https://github.com/zach-lewis/fantasybaseball_praw) was something I created as an attempt to get a leg up on identifying players that are demonstrating promise in terms of fantasy points, but may be prospects or have a low roster percentage across fantasy baseball leagues. It runs every morning at a set time and collects the names of players being discussed on Reddit and the number of times they're mentioned, stores the data in a JSON file, and sends a text with the top 15 players discussed that day. 

The program relies on players being mentioned by their full names, which at the time of writing I justified under the assumption that most new prospects or relative unknowns would be referred to by their full name. This could lead to potential skewing as popular players like Jacob deGrom or Javier Baez who are often referred to by last name won't have their mentions  accurately captured. However, I was comfortable with this omission (and even welcomed it) as the purpose of the tool is to identify players to pick up that have a large upside, and most people don't need a data collection tool to let them know that Jacob deGrom is an attractive player for their fantasy team. (https://www.pitcherlist.com/is-jacob-degrom-actually-good/)

After collecting a few months of data, I was curious to see how the data was shaping up, and evaluate how effective the tool is at surfacing valuable players.

## Analyzing Collected Data
### Data Loading and Preparation
Necessary libraries are loaded, and the latest mentions data from the locally stored JSON file

```python 
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import pybaseball
import seaborn as sns
from sklearn.linear_model import LinearRegression
```
```python
with open('data/mentions_data.json', 'r') as f:
    json_format = json.load(f)
    df = pd.read_json(json_format)
```
```python 
#Aggregate by Player and sort by total number of mentions
top_mentions = df.groupby('Players').Num_Mentions.sum()
top_mentions = top_mentions.reset_index().sort_values(by='Num_Mentions', ascending=False)
```
I was interested in seeing how the mentions data evolved over time, in addition to the full data. The following loops through the data for each player, reindexes, fills in any dates with no mentions with zeros, and calculates the cumulative mentions at each date.

```python
by_player = df.groupby('Players')
date_idx = pd.date_range('04-01-2021', datetime.today())

frames = []
for name, data in by_player:
   data.index = pd.DatetimeIndex(data.date)
   data = data.reindex(date_idx, fill_value = np.nan)
   data['Players'].fillna(value=list(set(data['Players'].dropna()))[0], inplace=True)
   data['Num_Mentions'].fillna(value=0, inplace=True)
   data['date'] = data.index
   data['Cumulative_Mentions'] = data["Num_Mentions"].cumsum()
   data.reset_index(inplace=True, drop=True)
   frames.append(data)
   
full = pd.concat(frames, ignore_index=True)
```

### Top Mentions to Date
An unfortunate result of the using names as unique identifiers is that a name like Will Smith shoots to the top of mentions. There's currently two prominent Will Smiths in the MLB - a catcher for the Dodgers and a relief pitcher for the Braves, who are both rostered in at least 85% of leagues. Given the data source is unstructured text from Reddit comments, it's a nuance I'm willing to take for a shot at winning back my league buy-in.

Of the top 15 players mentioned so far this year, I've picked up Trevor Rogers, Ty France, Robbie Ray, and Rich Hill at varying points. Rogers and France are newer players having solid seasons, while Robbie Ray and Rich Hill are more seasoned pitchers who have seen a bit of an unexpected resurgence in performance this year.

```python
graph = px.bar(top_mentions.iloc[:15, :], 'Players', 'Num_Mentions',
                hover_name='Players', hover_data=['Num_Mentions'], text='Num_Mentions')
iplot(graph)
```

{% include top_mentions.html %}

### Mentions Over Time
When looking at mentions over time, you can see how certain players have been discussed since the start of the season such as Rogers and France, while others show up during May/June like Luis Garcia and Rich Hill. It seems to generally follow the performance of the players (sans Luis Castillo who is a perennial frustration) - for instance, Nate Lowe begins high on the mentions with an explosive start to the year, but the mentions drop after his stats normalized. 

```python
full['date_str'] = full.date.astype(str)
full.sort_values(by='date_str', ascending=True, inplace=True)

graph_df = full.copy()
graph_frames = []
for _, data in graph_df.groupby('date_str'):
    data.sort_values(by='Cumulative_Mentions', ascending=False, inplace=True)
    graph_frames.append(data)
    
graph_df = pd.concat(graph_frames, ignore_index=True)

graph_df.rename({'Cumulative_Mentions' : 'Total Mentions', 
                   'date_str' : 'Date'}, axis='columns', inplace=True)

g = px.bar(graph_df, 'Players', 'Total Mentions',
                 text = 'Total Mentions', orientation = 'v', 
                 title='Top 15 Players Total Mentions in r/fantasybaseball',
                 animation_frame='Date', animation_group='Players',
                 range_y=[0, graph_df['Total Mentions'].max()+50])
g.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
g.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1500
g.update_xaxes(range = (-0.5,14.5))

iplot(g, auto_play=False, animation_opts = {'easing' : 'elastic', 'redraw' : False})
```

{% include top_ment_auto.html %}

## Evaluating the Relevance of Mentions with Statcast Data
While seeing the top mentions trends and how they've evolved is interesting, it's worth further investigation into whether or not mentions are correlated with the fantasy points a players accounts for, and therefore worth rostering. In place of scraping the ESPN website for my specific league to return fantasy points for each player, I utilized an approach that converts season stats obtained through pybaseball into fantasy points based on my particular leagues rules. (pybaseball is a great package that scrapes Baseball Reference, FanGraphs, and Baseball Savant https://github.com/jldbc/pybaseball)

```python
season_batting = pybaseball.batting_stats(2021)
season_pitching = pybaseball.pitching_stats(2021)

#A nuance of the praw_mod is it relies on joining lowercase names, and then converting them to proper case. This is 
#fine for most names, but deGrom needs to have his name converted in the pitching stats for a join that occurs later.
season_pitching.iloc[season_pitching[season_pitching.Name == 'Jacob deGrom'].index, 2] = 'Jacob Degrom'
```

For the purpose of the analysis, I created couple small functions to convert stats to fantasy points, and then plot a simple regression to evaluate the relationship between mentions and fantasy scores. 

```python
def convert_stats(df_row, points_dict):
    """Function to convert MLB stats to Fantasy Points based on League Scoring Rules"""
    return sum([df_row[k] * points_dict[k] for k in points_dict.keys()])

def graph_correlation(frame, x_col, y_col, hover_col, title):
    """
    Returns Plotly Scatter Plot with Line of Best Fit. Similar to Seaborn regplot, but with
    plotly interactivity to view player names
    """
    model = LinearRegression()
    x, y = frame[x_col].values.reshape(-1, 1), frame[y_col].values
    model.fit(x, y)
    b, m = model.intercept_, model.coef_

    reg = go.Scatter(x=frame[x_col], y=b + m * frame[x_col], name='Regression',
                         line=dict(color='firebrick', width=1), mode='lines')
    g = px.scatter(frame, x=x_col, y=y_col, hover_data=[hover_col],
                  title=f'{title} (R-Squared: {model.score(x,y):.3f})')
    g.add_trace(reg)
    return g

#Dictionaries Storing Fantasy Point Values per stat, customizable based on specific league rules. 
batting_scoring = {
    'R' : 1,
    'RBI' : 1,
    'TB' : 1,
    'BB' : 1, 
    'SO' : -1,
    'SB' : 1
}

pitching_scoring = {
    'IP' : 3,
    'H' : -1,
    'ER' : -2,
    'BB' : -1,
    'SO' : 1,
    'W' : 5,
    'L' : -5,
    'SV' : 5
}

#Convert Hits to Total Bases and apply points conversions
season_batting['TB'] = season_batting['1B'] + season_batting['2B']*2 + season_batting['3B']*3 + season_batting['HR']*4
season_batting['fantasy_bat'] = season_batting.apply(lambda row: convert_stats(row, batting_scoring), axis=1)
season_pitching['fantasy_pitch'] = season_pitching.apply(lambda row: convert_stats(row, pitching_scoring), axis=1)
```

### Position Players

```python
bat_corr = season_batting[['Name', 'fantasy_bat']].merge(top_mentions, left_on='Name', right_on='Players',
                                                           how='inner').drop('Players', axis=1)
iplot(graph_correlation(bat_corr, 'Num_Mentions', 'fantasy_bat', 'Name', 'Batting Correlation - All Players'))
```

{% include corr_1.html %}

For all players in the data set, it appears the number of mentions on reddit is a fairly poor predictor of fantasy points, though there is a slight positive correlation. However, this data set includes perrenial all-stars that I expected to skew the results, and likely aren't available to roster anyways. To solve for this and remove some of the noise, filtering based on Average Draft Pick - a measure of how early a player is drafted in all fantasy leagues - is a good measure of general popularity and is available for all leagues at https://www.fantasypros.com/mlb/adp/overall

```python
adp = pd.read_csv('data/2021_ADP_Rankings.csv')
#Select only Players drafted on average 250th or later
adp_filt = adp[adp.Rank >= 250]

no_stars = bat_corr[(bat_corr.Name.isin(adp_filt.Player))]
iplot(graph_correlation(no_stars, 'Num_Mentions', 'fantasy_bat', 'Name', 'Batting Correlation - No Star Players'))
```

{% include corr_2.html %}

After removing the players with a low ADP, the correlation improves a bit - it is still not an incredibly strong predictor, but it does a better job at estimating player value when looking at players drafted later. Now let's look at pitchers and see if the behavior is different.

### Pitchers
```python
pitch_corr = season_pitching[['Name', 'fantasy_pitch']].merge(top_mentions, left_on='Name', right_on='Players',
                                                           how='inner').drop('Players', axis=1)
iplot(graph_correlation(pitch_corr, 'Num_Mentions', 'fantasy_pitch', 'Name', 'Pitching Correlation - All Players' ))
```

{% include corr_3.html %}

Basically no statistical significance for pitchers at first glance, but lets take a look at how that changes when removing players with a lower ADP. Popular pitchers can account for a significant amount of points on a team, and the good ones are usually not available once the season gets started - however there are always a few breakouts or resurgences that could be nabbed off waivers.

```python
no_stars_p = pitch_corr[(pitch_corr.Name.isin(adp_filt.Player))]
iplot(graph_correlation(no_stars_p, 'Num_Mentions', 'fantasy_pitch', 'Name', 'Pitching Correlation - No Star Players'))
```
{% include corr_4.html %}

There's a significant improvement when looking at the pitchers that went later in the draft (or were unrostered), resulting in a fairly positive correlation for points in relation to the number of mentions. If a pitcher is getting lots of buzz on reddit, it seems worth an investigation into picking them up. 

Lastly, let's look at the whole picture of players with an ADP above 250th.

### All Players
```python
full = pd.concat([no_stars, no_stars_p]).fillna(0)
full['Fantasy Points'] = full.fantasy_bat + full.fantasy_pitch
iplot(graph_correlation(full, 'Num_Mentions', 'Fantasy Points', 'Name', 'Full Correlation - No Star Players'))
```
{% include corr_5.html %}

Based on the previous two graphs, it appears that the number of fantasy points increases sharply between 0-500 mentions, and then starts to taper off. Out of curiosity, let's see if a polynomial regression fits the data any better.

```python
sns.set()
fig, ax = plt.subplots(1,2, figsize = (16,10), sharey=True)
sns.regplot(x='Num_Mentions', y='Fantasy Points', data=full, order=2, ci=False, ax=ax[0])
ax[0].set_title("2nd Degree Polynomial")
sns.regplot(x='Num_Mentions', y='Fantasy Points', data=full, order=3, ci=False, ax=ax[1])
ax[1].set_title("3rd Degree Polynomial")
```

PLACEHOLDER

## Conclusion
When looking at the entire universe of players, there is not a strong correlation between the number of mentions collected from Reddit and their fantasy points. This could be driven by a variety of factors:
- The increased likelihood that they will be mentioned by last name only and missed by the tool
- Less likely to be discussed in relation to their trade prospects - most people aren't asking Reddit's opinion on trading for someone like Gerrit Cole as their value is more or less obvious, while lesser known player's values often gets discussed
- High value all-stars are less likely to be discussed in depth after a solid outing than middle/lower-tier players, while the consistent performance from someone unexpected tends to get more noise as owners look to roster a good value.

However, when restricted to only players drafted 250th or later as a proxy for general popularity / roster %, the correlation improves. Position players seem to follow a pattern of high mentions during hot streaks, which may normalize in the long run, while pitchers with high mentions tend to have a better long term value proposition. 

While not a perfect prediction of player value, mentions on Reddit appear to be a decent signal for identifying players worth researching for your team. For me, the best value of the tool is that it aggregates comments and provides a daily list of top players, saving me time skimming r/fantasybaseball which I can put into more targeted research and making an informed decision. 

So far this season I've picked up Trevor Rogers, Ty France, Rich Hill, Robbie Ray, Adolis Garcia, and Raimel Tapia after they were top mentions - all of whom have provided solid value to date. Whether or not that will contribute to my final league standings remains to be seen, but overall I am pleased with the tool and the edge it has given me in sourcing unrostered players with an upside.

