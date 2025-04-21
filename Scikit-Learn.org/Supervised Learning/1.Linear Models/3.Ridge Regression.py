from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB']]
y = df["CO2EMISSIONS"]
# Sample data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Always scale for Ridge!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Ridge Regression with alpha=1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print("R² score:", ridge.score(X_test_scaled, y_test))
print("Coefficients:", ridge.coef_)

coefficients = ridge.coef_
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