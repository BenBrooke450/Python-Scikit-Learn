


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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





X = df[['age', 'bmi', 'children', 'sex_female', 'sex_male',
       'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']]

y = df['charges']







# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (fit is done in place, no need to attach to variable)
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print(f'R-squared: {r2}')
#R-squared: 0.7835929767120723

"""
Imagine you're trying to predict house prices (y_test) based
       on features like square footage, number of rooms, etc.
       If your model's R-squared value is 0.7836, it means that
       78.36% of the variation in house prices can be explained
       by the features you've included in the model. The remaining
       21.64% could be due to other factors not included in the
       model (e.g., location, market trends, etc.).

In Summary:
R² = 1: Perfect prediction.
R² = 0: The model doesn't improve upon simply predicting the average.
0 < R² < 1: The model explains a portion of the variability in the data (for your case, 78.36%).
"""



reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.5306427503736921


"""
Conclusion:
Use model.score(X_test, y_test) if you want a quick
       way to evaluate the model’s performance with minimal code.

Use r2_score(y_test, y_pred) if you prefer more
       control over the prediction process or if
       you need to manipulate the predictions before evaluating them.
"""

