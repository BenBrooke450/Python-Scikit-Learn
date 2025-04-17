from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score':    [50, 55, 65, 70, 68, 75, 78, 85, 87, 95]
}
# Create DataFrame
df = pd.DataFrame(data)
X = df[["Hours_Studied"]]
y = df["Exam_Score"]
reg = linear_model.LinearRegression()
reg.fit(X, y)
print(reg.coef_)
#[4.60606061]
gradient = reg.coef_
print(reg.intercept_)
#47.466666666666676
intercept = reg.intercept_

# X is a Dataframe, we need to check it to either a series or an array
x = np.array(X)
x_line = np.linspace(min(x), max(x), 100)
y_line = gradient * x_line + intercept

plt.scatter(X,y)
plt.plot(x_line, y_line, color='red')



data = {
    'Hours_Studied': [
        4.3745, 9.9507, 7.7320, 6.5987, 1.1560, 1.1560, 0.5808, 8.6618, 6.0112, 7.0807,
        0.2058, 9.6991, 8.3244, 2.1234, 1.8182, 1.8340, 3.0424, 5.2476, 4.3195, 2.9123,
        6.1185, 1.3949, 2.9214, 3.6636, 4.5607, 7.8518, 1.9967, 5.1423, 5.9241, 0.4645,
        6.0754, 1.7052, 9.6991, 8.3244, 2.1234, 1.8182, 1.8340, 3.0424, 5.2476, 4.3195,
        2.9123, 6.1185, 1.3949, 2.9214, 3.6636, 4.5607, 7.8518, 1.9967, 5.1423, 5.9241,
        3.1185, 3.3949, 4.9214, 3.6636, 4.5607, 7.8518, 1.9967, 5.1423, 5.9241, 0.4645
    ],
    'Exam_Score': [
        67.54, 92.38, 88.99, 81.83, 51.04, 49.03, 46.80, 84.61, 78.33, 83.79,
        47.51, 94.56, 87.76, 54.93, 58.13, 52.70, 61.06, 68.55, 65.21, 61.08,
        75.71, 53.40, 60.66, 65.83, 66.37, 85.33, 58.97, 70.13, 73.96, 48.26,
        73.40, 56.40, 94.56, 87.76, 54.93, 58.13, 52.70, 61.06, 68.55, 65.21,
        61.08, 75.71, 53.40, 60.66, 65.83, 66.37, 85.33, 58.97, 70.13, 73.96,
        73.71, 52.41, 62.63, 64.83, 62.37, 83.31, 50.94, 70.11, 63.96, 29.26
    ]
}
df = pd.DataFrame(data)
X = df[["Hours_Studied"]]
y = df["Exam_Score"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=13)

#Train the model
model = LinearRegression()
model.fit(X_train,y_train)

#Now we have trained the model, we can X_test to test how well the y_pred == y_test
y_pred = model.predict(X_test)
print(y_pred)

#Let's compare the predicted values to the actual values
plt.scatter(y_test,y_pred)

from sklearn.metrics import mean_squared_error, r2_score
#r2_score and score are the same when applying to a LR
test = mean_squared_error(y_test,y_pred)
print("This tells us the unexplained variance between the predicted and actual values")
print("MSE:",test)
print("\nThis is the r_score/score of the input values X and output values y")
print(model.score(X,y))
print(f"\n{model.score(X,y)} This score clearly indicates a high correlation")



# X is a Dataframe, we need to check it to either a series or an array
x = np.array(X)
x_line = np.linspace(min(x), max(x), 100)
y_line = gradient * x_line + intercept
plt.scatter(X,y)
plt.plot(x_line, y_line, color='red')



print(X.shape)
print(y.shape)



url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)

