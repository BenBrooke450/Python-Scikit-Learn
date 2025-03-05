import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("/Users/benjaminbrooke/Library/Mobile Documents/com~apple~CloudDocs/housing[1].csv")

df = pd.DataFrame(df)

print(df)
"""
       longitude  latitude  ...  median_house_value  ocean_proximity
0        -122.23     37.88  ...            452600.0         NEAR BAY
1        -122.22     37.86  ...            358500.0         NEAR BAY
2        -122.24     37.85  ...            352100.0         NEAR BAY
3        -122.25     37.85  ...            341300.0         NEAR BAY
4        -122.25     37.85  ...            342200.0         NEAR BAY
...          ...       ...  ...                 ...              ...
20635    -121.09     39.48  ...             78100.0           INLAND
20636    -121.21     39.49  ...             77100.0           INLAND
20637    -121.22     39.43  ...             92300.0           INLAND
20638    -121.32     39.43  ...             84700.0           INLAND
20639    -121.24     39.37  ...             89400.0           INLAND

[20640 rows x 10 columns]
"""

desc = df.describe()
csv_output = desc.to_csv()
print(csv_output)
"""
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value
count,20640.0,20640.0,20640.0,20640.0,20433.0,20640.0,20640.0,20640.0,20640.0
mean,-119.56970445736432,35.63186143410853,28.639486434108527,2635.7630813953488,537.8705525375618,1425.4767441860465,499.5396802325581,3.8706710029069766,206855.81690891474
std,2.003531723502581,2.1359523974571117,12.585557612111637,2181.615251582787,421.3850700740322,1132.4621217653375,382.3297528316099,1.8998217179452732,115395.6158744132
min,-124.35,32.54,1.0,2.0,1.0,3.0,1.0,0.4999,14999.0
25%,-121.8,33.93,18.0,1447.75,296.0,787.0,280.0,2.5633999999999997,119600.0
50%,-118.49,34.26,29.0,2127.0,435.0,1166.0,409.0,3.5347999999999997,179700.0
75%,-118.01,37.71,37.0,3148.0,647.0,1725.0,605.0,4.74325,264725.0
max,-114.31,41.95,52.0,39320.0,6445.0,35682.0,6082.0,15.0001,500001.0
"""

df = df.dropna()

X = df[["total_bedrooms"]]
y = df["median_income"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=13)

lr = LinearRegression()
lr.fit(X_train,y_train)

print(lr.score(X_train, y_train))

print(lr.score(X_test, y_test))

