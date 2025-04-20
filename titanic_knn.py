import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Veri setini oku
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# İlk 5 satırı gör
print(df.head())

# Veri tipi ve eksik veri kontrolü
print(df.info())
print(df.isnull().sum())

# Cabin sütunu çok fazla eksik veri içerdiği için çıkarılıyor
df.drop("Cabin", axis=1, inplace=True)

# Age sütunu eksikse median ile doldur
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked sütunu eksikse en sık görülen değer (mode) ile doldur
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

label = LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])  # male = 1, female = 0
df["Embarked"] = label.fit_transform(df["Embarked"])

# Modelde kullanılmayacak metin sütunlarını çıkar
df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
