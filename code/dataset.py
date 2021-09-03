import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def load_data():
    data = pd.read_csv('../titanic_train.csv')

    data = data.loc[data.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    data[['Age']] = SimpleImputer(strategy="mean").fit_transform(data[['Age']])
    data['Age'] = pd.cut(data['Age'], bins=[0,12,20,40,120], labels=['Children', 'Teenage', 'Adult', 'Elder'])
    data['Fare'] = pd.qcut(data['Fare'], q=5, labels=['a', 'b', 'c', 'd', 'e'])
    data[['Sex', 'Embarked', 'Age', 'Fare']] = OrdinalEncoder().fit_transform(data[['Sex', 'Embarked', 'Age', 'Fare']])

    X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    y = data.Survived

    return X, y