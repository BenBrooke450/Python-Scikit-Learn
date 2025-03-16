import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import kagglehub
from IPython.display import display




#path = kagglehub.dataset_download("abhishek14398/salary-dataset-simple-linear-regression")
#print(path)

df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/abhishek14398/salary-dataset-simple-linear-regression/versions/1/Salary_dataset.csv")

df = pd.DataFrame(df)

display(df)

y = df["YearsExperience"]
X = df[['Salary']]

reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.9569566641435086

plt.scatter(X,y)

#plt.show()







##############################################


import seaborn as sns

sns.regplot(x = X, y = y)

#ci = Confidence Interval
#Or sns.regplot(x = X, y = y, ci=None)

#plt.show()

##############################################











from sklearn.model_selection import train_test_split

X = np.array(df["YearsExperience"]).reshape(-1, 1)
y = np.array(df[['Salary']]).reshape(-1, 1)

# Separating the data into independent and dependent variables

# Converting each dataframe into a numpy array

# since each dataframe contains only one column
df.dropna(inplace=True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Splitting the data into training and testing data
regr_test = LinearRegression()

regr_test.fit(X_train, y_train)

print(regr_test.score(X_test, y_test))
#0.9423541247498525











from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = regr_test.predict(X_test)

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
# squared True returns MSE value, False returns RMSE value.

mse = mean_squared_error(y_true=y_test, y_pred=y_pred)  # default=True

print("MAE:", mae)
#MAE: 3510.793448255886

print("MSE:", mse)
#MSE: 16189089.646805592


