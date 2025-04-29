import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]
# Load the dataset
df = pd.read_csv(url, names=column_names)
print(df.head())

print(df.columns)

df.info()

"""ca = pd.Series(df["ca"])
thal = pd.Series(df["thal"])
for i,x in enumerate(ca):
    try:
        ca[i] = float(x)
    except:
        ca[i] = 0.0
for i,x in enumerate(thal):
    try:
        thal[i] = float(x)
    except:
        thal[i] = 0.0
df['ca'] = ca
df['thal'] = thal
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)"""

df['ca'] = pd.to_numeric(df['ca'], errors='coerce').fillna(0.0)
df['thal'] = pd.to_numeric(df['thal'], errors='coerce').fillna(0.0)
df.info()

from sklearn.linear_model import LogisticRegression
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop(columns = ['num'],axis = 1)
print(x)

y = df["num"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create scaler
scaler = StandardScaler()
    
# Fit and transform
X_scaled_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train the model
model = LogisticRegression(max_iter = 1000000)
from sklearn.preprocessing import StandardScaler
model.fit(X_scaled_train, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)





conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

import plotly.express as px
from sklearn.metrics import confusion_matrix
import pandas as pd
# Get the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
labels = ['Negative', 'Positive']
fig = px.imshow(
    conf_matrix,
    text_auto=True,
    color_continuous_scale='Blues',
    title="Confusion Matrix"
)
fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
fig.show()

from sklearn.metrics import precision_score, recall_score, f1_score
# Calculate metrics
precision = precision_score(y_test, y_pred, average = None)
recall = recall_score(y_test, y_pred, average = None)
f1 = f1_score(y_test, y_pred, average = None)

print("Precision:", precision,"\n")
print("Recall:", recall,"\n")
print("F1 Score:", f1,"\n")


coef_df = pd.DataFrame(model.coef_, columns=X.columns)
coef_df['Class'] = model.classes_
coef_df = coef_df.set_index('Class').T  # transpose for better plotting
coef_df.plot(kind='barh', figsize=(10, 8))
plt.title('Logistic Regression Feature Impact by Class')
plt.xlabel('Coefficient Value')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()