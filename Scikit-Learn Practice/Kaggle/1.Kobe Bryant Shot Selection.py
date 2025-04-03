
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

first_2_seasons = df[df["season"] < 3]

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




first_2_seasons.to_csv('/Users/benjaminbrooke/Downloads/Kaggle Datasets/Kobe Bryant Shot Selection/first_2_seasons.csv', index=False)




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error








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


first_2_seasons['Shot_over_each_game']

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
#MAE: 4276.899491054204

print("MSE:", mse)
#MSE: 22042582.020160593







# Step 4: Plot the regression line over the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')  # Actual training data points
plt.plot(X_train, reg.predict(X_train), color='red', label='Linear regression line')  # Line of best fit

# Optionally, plot test predictions (test data + predictions)
plt.scatter(X_test, y_pred, color='green', label='Test data predictions')  # Test data predictions
plt.title('Linear Regression: Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
#plt.show()



print("Slope (m):", reg.coef_[0])
#Slope (m): 9449.962321455072

print("Intercept (b):", reg.intercept_)
#Intercept (b): 24848.203966523222

"""
Slope (m) = 5000: For each additional year of experience, the salary increases by 5000 units.

Intercept (b) = 35000: When years of experience is zero, the starting salary is 35000.
"""







