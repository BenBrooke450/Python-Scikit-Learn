

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")

print("Path to dataset files:", path)"""




df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/wenruliu/adult-income-dataset/versions/2/adult.csv")

df = pd.DataFrame(df)

print(df)



X = df[['hours-per-week']]

y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#0.7522776128569966



# Create a range of values to plot the decision boundary
xx = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)

# Predict probabilities (we use `predict_proba` to get probabilities for both classes)
probs = model.predict_proba(xx)[:, 1]  # We take the probability of class 1

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='black', marker='o', label='Data points')

# Plot the logistic regression curve (decision boundary)
plt.plot(xx, probs, color='red', label='Logistic Regression Curve')

# Add a horizontal line at 0.5 to show the decision boundary
plt.axhline(y=0.5, color='blue', linestyle='--', label='Decision Boundary')

# Format the plot
plt.xlabel('Feature X')
plt.ylabel('Probability')
plt.title('Logistic Regression with One Input Feature')
plt.legend()
plt.show()

