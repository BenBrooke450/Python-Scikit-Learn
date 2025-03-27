



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("kandij/diabetes-dataset")

print("Path to dataset files:", path)"""




df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/kandij/diabetes-dataset/versions/1/diabetes2.csv")

df = pd.DataFrame(df)

print(df)
"""
     Pregnancies  Glucose  ...  Age  Outcome
0              6      148  ...   50        1
1              1       85  ...   31        0
2              8      183  ...   32        1
3              1       89  ...   21        0
4              0      137  ...   33        1
..           ...      ...  ...  ...      ...
763           10      101  ...   63        0
764            2      122  ...   27        0
765            5      121  ...   30        0
766            1      126  ...   47        1
767            1       93  ...   23        0
"""

y = df['Outcome']
X = df[["Age","DiabetesPedigreeFunction","BMI","Insulin","SkinThickness","BloodPressure","Glucose","Pregnancies"]]


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(accuracy)
#0.7467532467532467

"""
0.7 indicates that there is a correlation between
    the X (typically refers to the input features or independent variables) and y 
    (is the target variable (dependent variable) that we are trying to predict.)
"""








from sklearn.metrics import precision_score, log_loss

precision = precision_score(y_test, y_pred)

print(f"Precision: {precision}")

#Precision: 0.6379310344827587

"""
Precision (also known as Positive Predictive Value) is 
    the proportion of positive predictions that are actually correct.
"""










"""
For log loss, you need to get the predicted probabilities 
(not the predicted class labels). Logistic Regression provides 
a method predict_proba() that returns the probability estimates for each class.

"""

all = model.predict_proba(X_test)[:]

print(len(all))
#154

print(all)
"""
[[0.72413342 0.27586658]
 [0.81155622 0.18844378]
 [0.88551454 0.11448546]
 [0.83645463 0.16354537]
 ...
 ...
 ...
"""


"""
predict_proba() returns a 2D array, where the 
    first column represents the probability of 
    class 0 (negative class), and the second column 
    represents the probability of class 1 (positive class). 
    We are interested in the second column, so we use [:, 1].
"""

y_pred_prob = model.predict_proba(X_test)[:, 1]

log_loss_value = log_loss(y_test, y_pred_prob)

print(f"Log Loss: {log_loss_value}")
#Log Loss: 0.5117112508521531

