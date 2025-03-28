


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Medical Cost Personal Datasets

import kagglehub

# Download latest version
#path = kagglehub.dataset_download("mirichoi0218/insurance")
#print(path)


df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/mirichoi0218/insurance/versions/1/insurance.csv")

df = pd.DataFrame(df)

display(df)
"""
      age     sex     bmi  children smoker     region      charges
0      19  female  27.900         0    yes  southwest  16884.92400
1      18    male  33.770         1     no  southeast   1725.55230
2      28    male  33.000         3     no  southeast   4449.46200
3      33    male  22.705         0     no  northwest  21984.47061
4      32    male  28.880         0     no  northwest   3866.85520
...   ...     ...     ...       ...    ...        ...          ...
1333   50    male  30.970         3     no  northwest  10600.54830
1334   18  female  31.920         0     no  northeast   2205.98080
1335   18  female  36.850         0     no  southeast   1629.83350
1336   21  female  25.800         0     no  southwest   2007.94500
1337   61  female  29.070         0    yes  northwest  29141.36030

"""

y = df['charges']
X = df[["age","sex","bmi","children","smoker","region"]]



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded_location = encoder.fit_transform(df[['sex', 'smoker', 'region']])

encoded_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(['sex', 'smoker', 'region']))

# Combine the encoded columns with the original DataFrame
df = pd.concat([df[['age', 'bmi', 'children','charges']], encoded_df], axis=1)

print(df.columns)
"""
Index(['age', 'bmi', 'children', 'charges', 'sex_female', 'sex_male',
       'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest'],
      dtype='object')
"""





X = df[['children',
       'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']]

#y = df['charges'] can`t use as this is a continuous column, we're looking for binary outcomes

y = df['sex_female']





# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(accuracy)
#0.47388059701492535

"""
An accuracy score of 0.474 (or roughly 47.4%) in a Logistic Regression model
       suggests that the model is performing worse than random guessing, especially
       if you have a binary classification task with balanced classes. It’s essential
       to analyze the situation further to understand why this is happening and what
       you can do to improve your model. Let’s break this down and explore a
       few possible reasons and steps to diagnose and improve the performance.
"""


print(conf_matrix)
"""
[[59 69]
 [72 68]]
"""


"""
True Negatives (TN) = 59
False Positives (FP) = 69
False Negatives (FN) = 72
True Positives (TP) = 68

What do these values mean?
True Negatives (TN): The model correctly predicted negative instances (class 0). Here, 59 instances were correctly classified as negative.
False Positives (FP): The model incorrectly predicted positive (class 1) when the true label was negative (class 0). Here, 69 instances were incorrectly predicted as positive.
False Negatives (FN): The model incorrectly predicted negative (class 0) when the true label was positive (class 1). Here, 72 instances were incorrectly predicted as negative.
True Positives (TP): The model correctly predicted positive instances (class 1). Here, 68 instances were correctly classified as positive.
"""

















