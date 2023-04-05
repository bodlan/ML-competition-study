import os

import numpy as np

import pandas as pd

from typing import List
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

temp_dir = os.path.join(os.path.dirname(__file__), "../temp")
(gender_submission, test_path, train_path) = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir)]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
gender_submission = pd.read_csv(gender_submission)

test_ids = test["PassengerId"]


def substrings_in_string(full_str: str, substrings: List[str]):
    for substring in substrings:
        if full_str.find(substring) != -1:
            return substring
    return np.nan


def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == "Dr":
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))
    data['Title'] = data.apply(replace_titles, axis=1)
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    data['Cabin'] = data['Cabin'].astype(str)
    data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    data['Family_Size'] = data['SibSp'] + data['Parch']
    data['Fare_per_Person'] = data['Fare'] / (data['Family_Size'] + 1)
    data = data.drop(["Ticket", "PassengerId", "Cabin", "Name"], axis=1)
    cols = ["SibSp", "Parch", "Fare", "Age", "Family_Size", "Fare_per_Person"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data.Embarked.fillna("U", inplace=True)
    return data


train = clean_data(train)
test = clean_data(test)

le = preprocessing.LabelEncoder()
cols = ["Sex", "Embarked", "Title", "Deck"]
for col in cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    print(le.classes_)
print(train.head())

y = train.pop("Survived")
x = train

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)
predictions = clf.predict(x_val)

print(accuracy_score(y_val, predictions))

submission_preds = clf.predict(test)
df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds,
                   })
df.to_csv("submission.csv", mode="w", index=False)
