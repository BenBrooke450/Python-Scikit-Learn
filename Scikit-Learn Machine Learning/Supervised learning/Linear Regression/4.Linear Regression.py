import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import kagglehub
from IPython.display import display




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


reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.7509130345985207















X = df[['age']]
y = df['charges']


reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.08940589967885804


"""
import seaborn as sns

sns.regplot(x = X, y = y)

#ci = Confidence Interval

sns.regplot(x = X, y = y, ci=None)

plt.show()
"""













X = df[['bmi']]
y = df['charges']

reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.03933913991786264











X = df[['children']]
y = df['charges']


reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.004623758854459203












X = df[['smoker_yes']]
y = df['charges']

reg = LinearRegression().fit(X,y)

reg_score = reg.score(X,y)

print(reg_score)
#0.6197648148218988


import seaborn as sns
sns.regplot(x = X, y = y)

#ci = Confidence Interval
sns.regplot(x = X, y = y, ci=None)

plt.show()


