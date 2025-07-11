

import pandas as pd

# Load dataset (assume it's a CSV)
df = pd.read_csv("/Users/benjaminbrooke/Desktop/Data/sms+spam+collection/SMSSpamCollection.csv", encoding="latin-1")

df = pd.DataFrame(df)

df.columns = ["Info"]

#df[['Spam', 'Email']] = df[''].str.split(' ', expand=True)

print(df.columns)

print(df["Info"])


df[['Spam', 'Email']] = df['Info'].str.split('\t',n=1,expand=True)

df = df.drop("Info",axis=1)

print(df)







from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df['Spam'] = df['Spam'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['Email'], df['Spam'], test_size=0.2)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)




from sklearn.metrics import accuracy_score

predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
#Accuracy: 0.9838565022421525