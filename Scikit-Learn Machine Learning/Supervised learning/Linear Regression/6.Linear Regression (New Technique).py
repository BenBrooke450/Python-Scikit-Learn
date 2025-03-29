


import numpy as np
import pandas as pd

df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/mirichoi0218/insurance/versions/1/insurance.csv")

df = pd.DataFrame(df)

df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
dtypes: float64(2), int64(2), object(3)
memory usage: 73.3+ KB
"""



print(df.describe())
"""
               age          bmi     children       charges
count  1338.000000  1338.000000  1338.000000   1338.000000
mean     39.207025    30.663397     1.094918  13270.422265
std      14.049960     6.098187     1.205493  12110.011237
min      18.000000    15.960000     0.000000   1121.873900
25%      27.000000    26.296250     0.000000   4740.287150
50%      39.000000    30.400000     1.000000   9382.033000
75%      51.000000    34.693750     2.000000  16639.912515
max      64.000000    53.130000     5.000000  63770.428010
"""



import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


print(df.age.describe())
"""
count    1338.000000
mean       39.207025
std        14.049960
min        18.000000
25%        27.000000
50%        39.000000
75%        51.000000
max        64.000000
Name: age, dtype: float64
"""





print(df.smoker.value_counts())
"""
smoker
no     1064
yes     274
Name: count, dtype: int64
"""









print(df.charges.corr(df.age))
#0.2990081933306478


print(df.charges.corr(df.children))
#0.06799822684790487


print(df.charges.corr(df.bmi))
#0.19834096883362887


df['smoker'] = np.where(df['smoker'] == 'yes', 1, 0)

df['sex'] = np.where(df['sex'] == 'male', 1, 0)

#print(df.to_string())
"""
      age  sex     bmi  children  smoker     region       charges
0      19    0  27.900         0       0  southwest  16884.924000
1      18    1  33.770         1       0  southeast   1725.552300
2      28    1  33.000         3       0  southeast   4449.462000
3      33    1  22.705         0       0  northwest  21984.470610
4      32    1  28.880         0       0  northwest   3866.855200
5      31    0  25.740         0       0  southeast   3756.621600
"""

df = df.drop(columns = ["region"])


print(df.corr())
"""
               age       sex       bmi  children    smoker   charges
age       1.000000 -0.020856  0.109272  0.042469 -0.025019  0.299008
sex      -0.020856  1.000000  0.046371  0.017163  0.076185  0.057292
bmi       0.109272  0.046371  1.000000  0.012759  0.003750  0.198341
children  0.042469  0.017163  0.012759  1.000000  0.007673  0.067998
smoker   -0.025019  0.076185  0.003750  0.007673  1.000000  0.787251
charges   0.299008  0.057292  0.198341  0.067998  0.787251  1.000000
"""




df_non_smokers = df[df["smoker"]==0]
print(df_non_smokers.head(10))
"""
    age  sex     bmi  children  smoker      charges
1    18    1  33.770         1       0   1725.55230
2    28    1  33.000         3       0   4449.46200
3    33    1  22.705         0       0  21984.47061
4    32    1  28.880         0       0   3866.85520
5    31    0  25.740         0       0   3756.62160
6    46    0  33.440         1       0   8240.58960
7    37    0  27.740         3       0   7281.50560
8    37    1  29.830         2       0   6406.41070
9    60    0  25.840         0       0  28923.13692
10   25    1  26.220         0       0   2721.32080
"""

print(df_non_smokers.describe())
"""
               age          sex          bmi     children  smoker       charges
count  1064.000000  1064.000000  1064.000000  1064.000000  1064.0   1064.000000
mean     39.385338     0.485902    30.651795     1.090226     0.0   8434.268298
std      14.083410     0.500036     6.043111     1.218136     0.0   5993.781819
min      18.000000     0.000000    15.960000     0.000000     0.0   1121.873900
25%      26.750000     0.000000    26.315000     0.000000     0.0   3986.438700
50%      40.000000     0.000000    30.352500     1.000000     0.0   7345.405300
75%      52.000000     1.000000    34.430000     2.000000     0.0  11362.887050
max      64.000000     1.000000    53.130000     5.000000     0.0  36910.608030
"""



from sklearn.linear_model import LinearRegression

model =  LinearRegression()

inputs = df_non_smokers[["age"]]
target = df_non_smokers.charges

print("input: ", inputs.shape)
print("target: ", target.shape)

#input:  (1064, 1)
#target:  (1064,)


model.fit(inputs,target)

prediction = model.predict(inputs)

print(prediction)
"""
[2719.0598744  5391.54900271 6727.79356686 ... 2719.0598744  2719.0598744
 3520.80661289]
"""

print(target)
"""
1        1725.55230
2        4449.46200
3       21984.47061
4        3866.85520
5        3756.62160
           ...     
1332    11411.68500
1333    10600.54830
1334     2205.98080
1335     1629.83350
1336     2007.94500
"""


print(model.coef_)
#[267.24891283]

print(model.intercept_)
#-2091.420556565021












