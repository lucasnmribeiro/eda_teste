import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

titanic_data = pd.read_csv("titanic.csv")

print(titanic_data.head())

print(titanic_data.describe())

print(titanic_data.isnull().sum())

titanic_data.hist(bins=20, figsize=(14, 10))
plt.show()

categorical_features = ['Sex', 'Pclass', 'Embarked']
for feature in categorical_features:
    sns.countplot(x=feature, data=titanic_data)
    plt.show()

sns.barplot(x="Sex", y="Survived", data=titanic_data)
plt.show()

sns.barplot(x="Pclass", y="Survived", data=titanic_data)
plt.show()

titanic_data['Age'].dropna().hist(bins=20, figsize=(14, 10))
plt.show()

titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())

print(titanic_data.isnull().sum())

numeric_features = titanic_data.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

print("EDA concluída com sucesso!")
