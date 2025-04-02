

import pandas as pd
import plotly.express as px


# https://www.kaggle.com/competitions/kobe-bryant-shot-selection/data


df = pd.read_csv('/Users/benjaminbrooke/Downloads/Kaggle Datasets/Kobe Bryant Shot Selection/data.csv')

df = pd.DataFrame(df)

print(df)
"""
0              Jump Shot          Jump Shot  ...       POR        1
1              Jump Shot          Jump Shot  ...       POR        2
2              Jump Shot          Jump Shot  ...       POR        3
3              Jump Shot          Jump Shot  ...       POR        4
4      Driving Dunk Shot               Dunk  ...       POR        5
...                  ...                ...  ...       ...      ...
30692          Jump Shot          Jump Shot  ...       IND    30693
30693           Tip Shot           Tip Shot  ...       IND    30694
30694  Running Jump Shot          Jump Shot  ...       IND    30695
30695          Jump Shot          Jump Shot  ...       IND    30696
30696          Jump Shot          Jump Shot  ...       IND    30697
"""

print(df.columns)
"""
Index(['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat',
       'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs',
       'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag',
       'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
       'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id'],
      dtype='object')
"""

#df[["season_year","season"]] = df["season"].str.split("-",expand=True)

#df.to_csv('/Users/benjaminbrooke/Downloads/Kaggle Datasets/Kobe Bryant Shot Selection/data.csv', index=False)

print(df)


first_2_seaons = df[df["season"]<3]

fig = px.bar(x=first_2_seaons["season"], y=first_2_seaons["combined_shot_type"])

#fig.show()











