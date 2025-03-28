


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split




"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("ignacioazua/life-expectancy")

print("Path to dataset files:", path)"""




df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/ignacioazua/life-expectancy/versions/1/life_expectancy.csv")

df = pd.DataFrame(df)

print(df.columns)

df["Country_Rank"] = df["Country"].rank(ascending=False)

df = df.sort_values(by = ["Sum of Life Expectancy  (both sexes)"])

print(df.to_string())


X = df["Country"]
y = df["Sum of Life Expectancy  (both sexes)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=1)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)

# Visualize the training data and the polynomial fit
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_train_poly, y_train_pred, color='red', label='Polynomial fit')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

