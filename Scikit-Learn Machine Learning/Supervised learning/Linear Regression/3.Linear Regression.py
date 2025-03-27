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

X = df[["YearsExperience"]]
y = df['Salary']

reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.9569566641435086




print(X.shape)
#(30, 1)

print(y.shape)
#(30,)












from sklearn.model_selection import train_test_split


# since each dataframe contains only one column
df.dropna(inplace=True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.metrics import mean_absolute_error, mean_squared_error

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



