
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


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

df["season"] = df["season"].astype(int)

first_2_seasons = df[df["season"] < 3]

print(len(first_2_seasons))
#5473

fig = px.bar(x=first_2_seasons["season"], y=first_2_seasons["combined_shot_type"])

#fig.show()





first_2_seasons['Shot_over_each_game'] = first_2_seasons.groupby('game_id').transform('size')

print(first_2_seasons)
"""
             action_type combined_shot_type  ...  season_year  Shot_over_each_game
0              Jump Shot          Jump Shot  ...         2000                   11
1              Jump Shot          Jump Shot  ...         2000                   11
2              Jump Shot          Jump Shot  ...         2000                   11
3              Jump Shot          Jump Shot  ...         2000                   11
4      Driving Dunk Shot               Dunk  ...         2000                   11
...                  ...                ...  ...          ...                  ...
30692          Jump Shot          Jump Shot  ...         1999                   27
30693           Tip Shot           Tip Shot  ...         1999                   27
30694  Running Jump Shot          Jump Shot  ...         1999                   27
30695          Jump Shot          Jump Shot  ...         1999                   27
30696          Jump Shot          Jump Shot  ...         1999                   27
"""





fig_2 = px.box(x=first_2_seasons["season"], y=first_2_seasons['Shot_over_each_game'])

#fig_2.show()






first_2_seasons = first_2_seasons.drop(columns =
                                       {"shot_zone_basic","lat","loc_x"
                                           ,"loc_y","lon","team_id",
                                        "game_event_id"})




#first_2_seasons.to_csv('/Users/benjaminbrooke/Downloads/Kaggle Datasets/Kobe Bryant Shot Selection/first_2_seasons.csv', index=False)






from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error



first_2_seasons = first_2_seasons.dropna()


print("FIND NULL VALUES:",first_2_seasons["Shot_over_each_game"].isna())
"""
FIND NULL VALUES: 1        False
2        False
3        False
4        False
5        False
         ...  
30691    False
30692    False
30694    False
30695    False
30696    False
"""


rows_with_nan = first_2_seasons[first_2_seasons.isnull().any(axis=1)]
print("THIS IS THE NULL:  ",rows_with_nan)
#THIS IS THE NULL:   Empty DataFrame







first_2_seasons.replace(" ", pd.NA, inplace=True)

first_2_seasons = first_2_seasons.dropna()

first_2_seasons = first_2_seasons.dropna(subset=['Shot_over_each_game'])







####### SCIKIT-LEARN ####################################################### SECTION 2

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded_location = encoder.fit_transform(df[["opponent"]])

encoded_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(["opponent"]))

# Combine the encoded columns with the original DataFrame
first_2_seasons = pd.concat([first_2_seasons, encoded_df], axis=1)

print(df.columns)
"""

Index(['action_type', 'combined_shot_type', 'game_id', 'minutes_remaining',
       'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance',
       'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_range',
       'team_name', 'game_date', 'matchup', 'opponent', 'shot_id',
       'season_year', 'Shot_over_each_game', 'opponent_ATL', 'opponent_BKN',
       'opponent_BOS', 'opponent_CHA', 'opponent_CHI', 'opponent_CLE',
       'opponent_DAL', 'opponent_DEN', 'opponent_DET', 'opponent_GSW',
       'opponent_HOU', 'opponent_IND', 'opponent_LAC', 'opponent_MEM',
       'opponent_MIA', 'opponent_MIL', 'opponent_MIN', 'opponent_NJN',
       'opponent_NOH', 'opponent_NOP', 'opponent_NYK', 'opponent_OKC',
       'opponent_ORL', 'opponent_PHI', 'opponent_PHX', 'opponent_POR',
       'opponent_SAC', 'opponent_SAS', 'opponent_SEA', 'opponent_TOR',
       'opponent_UTA', 'opponent_VAN', 'opponent_WAS'],
      dtype='object')
      """





first_2_seasons.replace(" ", pd.NA, inplace=True)

first_2_seasons = first_2_seasons.dropna()

first_2_seasons = first_2_seasons.dropna(subset=['Shot_over_each_game'])





X = first_2_seasons[['opponent_ATL','opponent_BKN','opponent_BOS',
                     'opponent_CHA','opponent_CHI','opponent_CLE',
                     'opponent_DAL','opponent_DEN','opponent_DET',
                     'opponent_GSW','opponent_HOU','opponent_IND',
                     'opponent_LAC','opponent_MEM','opponent_MIA',
                     'opponent_MIL','opponent_MIN','opponent_NJN',
                     'opponent_NOH','opponent_NOP','opponent_NYK',
                     'opponent_OKC','opponent_ORL','opponent_PHI',
                     'opponent_PHX','opponent_POR','opponent_SAC',
                     'opponent_SAS','opponent_SEA','opponent_TOR',
                     'opponent_UTA','opponent_VAN','opponent_WAS']]



y = first_2_seasons['Shot_over_each_game']



reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)





print(X.shape)

print(y.shape)











# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_pred = reg.predict(X_test)



mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
# squared True returns MSE value, False returns RMSE value.

mse = mean_squared_error(y_true=y_test, y_pred=y_pred)  # default=True



print("MAE:", mae)
#MAE: 4.208455138620898

print("MSE:", mse)
#MSE: 26.387335323980704


print("Slope (m):", reg.coef_[0])
#Slope (m): -2.17397866758449

print("Intercept (b):", reg.intercept_)
#Intercept (b): 21.351398022423428




############################################################## SECTION 3





first_2_seasons = first_2_seasons.drop(columns =['opponent_ATL','opponent_BKN','opponent_BOS',
                                                  'opponent_CHA','opponent_CHI','opponent_CLE',
                                                  'opponent_DAL','opponent_DEN','opponent_DET',
                                                  'opponent_GSW','opponent_HOU','opponent_IND',
                                                  'opponent_LAC','opponent_MEM','opponent_MIA',
                                                  'opponent_MIL','opponent_MIN','opponent_NJN',
                                                  'opponent_NOH','opponent_NOP','opponent_NYK',
                                                  'opponent_OKC','opponent_ORL','opponent_PHI',
                                                  'opponent_PHX','opponent_POR','opponent_SAC',
                                                  'opponent_SAS','opponent_SEA','opponent_TOR',
                                                  'opponent_UTA','opponent_VAN','opponent_WAS'])




first_2_seasons["dates_rank"] = (first_2_seasons.
                                 groupby(["game_date","opponent"]).cumcount() + 1)

score_per_game = first_2_seasons[first_2_seasons["dates_rank"] == 1]

fig_3 = px.bar(x=score_per_game["opponent"], y=score_per_game["Shot_over_each_game"])

fig_3.update_xaxes(categoryorder='total ascending')

#fig_3.show()



fig_4 = px.bar(x=score_per_game["opponent"],
               y=score_per_game["Shot_over_each_game"],color=score_per_game["combined_shot_type"])

fig_4.update_xaxes(categoryorder='total ascending')

#fig_4.show()







############################################################## SECTION 4






from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix



first_2_seasons["shot_zone_range_num"] = np.where(first_2_seasons["shot_zone_range"] == "Less Than 8 ft.",0
                                                  ,np.where(first_2_seasons["shot_zone_range"] == "8-16 ft.", 1
                                                            ,np.where(first_2_seasons["shot_zone_range"] == "16-24 ft.", 2
                                                                      ,np.where(first_2_seasons["shot_zone_range"] == "24+ ft.", 3,None)))
                                                  )
print(first_2_seasons)
"""
              action_type combined_shot_type  ...  dates_rank  shot_zone_range_num
1               Jump Shot          Jump Shot  ...           1                    1
2               Jump Shot          Jump Shot  ...           2                    2
3               Jump Shot          Jump Shot  ...           3                    2
4       Driving Dunk Shot               Dunk  ...           4                    0
5               Jump Shot          Jump Shot  ...           5                    1
...                   ...                ...  ...         ...                  ...
30691  Driving Layup Shot              Layup  ...          19                    0
30692           Jump Shot          Jump Shot  ...          20                    0
30694   Running Jump Shot          Jump Shot  ...          21                    2
30695           Jump Shot          Jump Shot  ...          22                    3
30696           Jump Shot          Jump Shot  ...          23                    0
"""



df_2 = first_2_seasons
print(df_2["action_type"].unique())
"""
['Jump Shot' 'Driving Dunk Shot' 'Layup Shot' 'Running Jump Shot'
 'Reverse Dunk Shot' 'Slam Dunk Shot' 'Driving Layup Shot'
 'Turnaround Jump Shot' 'Reverse Layup Shot' 'Tip Shot'
 'Running Hook Shot' 'Alley Oop Dunk Shot' 'Dunk Shot'
 'Alley Oop Layup shot' 'Running Dunk Shot' 'Driving Finger Roll Shot'
 'Running Layup Shot' 'Finger Roll Shot' 'Fadeaway Jump Shot'
 'Follow Up Dunk Shot' 'Hook Shot' 'Turnaround Hook Shot'
 'Driving Hook Shot']
"""


df_2["combined_shot_type_num"] = pd.factorize(df_2["combined_shot_type"])[0]

df_2["opponent_num"] = pd.factorize(df_2["opponent"])[0]

X = df_2[["combined_shot_type_num",
       'minutes_remaining', 'period', 'season', 'seconds_remaining',
       'shot_distance', "opponent_num"]]

y = df_2['shot_made_flag']




# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(accuracy)
#0.6093579978237215


print(conf_matrix)
"""
[[365 149]
 [210 195]]
"""



df_2 = df_2[(df_2["game_date"] == "2000-10-31") | (df_2["game_date"] == "2000-11-01")]

X = df_2[["combined_shot_type_num",
       'minutes_remaining', 'period', 'season', 'seconds_remaining',
       'shot_distance', "opponent_num"]]

df_2['predicted_class'] = model.predict(X)

print(df_2)

df_2.to_csv('/Users/benjaminbrooke/Downloads/Kaggle Datasets/Kobe Bryant Shot Selection/first_two_games_predict.csv', index=False)













