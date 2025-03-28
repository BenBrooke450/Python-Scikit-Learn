
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")

print("Path to dataset files:", path)"""




df = pd.read_csv("/Users/benjaminbrooke/.cache/kagglehub/datasets/wenruliu/adult-income-dataset/versions/2/adult.csv")

df = pd.DataFrame(df)

print(df)




import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt





from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded_location = encoder.fit_transform(df[['workclass', 'marital-status','race','gender','native-country']])

encoded_df = pd.DataFrame(encoded_location, columns=encoder.get_feature_names_out(['workclass', 'marital-status','race','gender','native-country']))

# Combine the encoded columns with the original DataFrame
df = pd.concat([df[['age','hours-per-week','income']], encoded_df], axis=1)

list1 = []
for x in df.columns:
    list1.append(x)

print(list1)

X = df[['hours-per-week', 'age','workclass_?', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay', 'marital-status_Divorced', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'gender_Female', 'gender_Male', 'native-country_?', 'native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']]

y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

scalar = MinMaxScaler(feature_range=(0,1))
X_train = scalar.fit_transform(X_train)

#Only X_train is fitted with the scaler.

X_test_scaled = scalar.transform(X_test)

"""
# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
X_train_scaled = scaler.fit_transform(X_train)
"""



knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(y_pred)
#['>50K' '<=50K' '<=50K' ... '<=50K' '<=50K' '>50K']



score = knn.score(X_test,y_test)
print(score)
#0.7876957723410789


cm = confusion_matrix(y_test,y_pred)
print(cm)
"""
[[6749  708]
 [1366  946]]
"""










