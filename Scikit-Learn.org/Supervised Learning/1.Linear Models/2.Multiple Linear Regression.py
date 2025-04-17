from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df)

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

print(df.columns)





df['FUELTYPE'] = pd.factorize(df['FUELTYPE'])[0]

#Now it's in numerical format
print(df['FUELTYPE'])

X = df[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS',
        'FUELTYPE', 'FUELCONSUMPTION_CITY',
       'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
       'FUELCONSUMPTION_COMB_MPG']]
y = df["CO2EMISSIONS"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X.to_string)

model = LinearRegression()

from sklearn.preprocessing import StandardScaler
# Create scaler
scaler = StandardScaler()
	
	# Fit and transform
X_scaled_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_scaled_train,y_train)



print ('Coefficients: ', model.coef_) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',model.intercept_)

coefficients = model.coef_
features = X.columns
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=True)
plt.figure(figsize=(8, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
plt.axvline(0, color='grey', linestyle='--')
plt.title('Feature Impact (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

X_scaled = scaler.fit_transform(X)
print(model.score(X_scaled,y))

X = df[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']]
y = df["CO2EMISSIONS"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
	
X_scaled_train = scaler.fit_transform(X_train)
model_2 = LinearRegression()
model_2.fit(X_scaled_train,y_train)
X_scaled = scaler.fit_transform(X)
print(model_2.score(X_scaled,y))

coefficients = model_2.coef_
features = X.columns
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=True)
plt.figure(figsize=(8, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
plt.axvline(0, color='grey', linestyle='--')
plt.title('Feature Impact (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()