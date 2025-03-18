

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split




"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("taseermehboob9/salary-dataset-of-business-levels")

print("Path to dataset files:", path)

"""


df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/taseermehboob9/salary-dataset-of-business-levels/versions/1/salary.csv")

df = pd.DataFrame(df)

print(df.columns)

X = df[["Level "]]
y = df["Salary"]


print(df["Level "].dtype)
#Index(['Position', 'Level ', 'Salary'], dtype='object')

df = df.rename(columns={'Level ': 'Level'})

print(df.columns)
#Index(['Position', 'Level', 'Salary'], dtype='object')


X = df[["Level"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

# Visualize the training data and the polynomial fit
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(np.sort(X_train, axis=0), model.predict(poly.transform(np.sort(X_train, axis=0))), color='red', label='Polynomial fit')
plt.xlabel('Feature')
plt.ylabel('Target')
#plt.show()



