import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)

temp_dir = os.path.join(os.path.dirname(__file__), "../temp")
(gender_submission, test_path, train_path) = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir)]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
combine = [train, test]
gender_submission = pd.read_csv(gender_submission)
print(train.columns.values)
print(train.head())
train.info()
print('_' * 40)
test.info()
print(train.describe())
print(train.describe(include=['O']))

pclass_surv = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() \
    .sort_values(by='Survived', ascending=False)
print(pclass_surv)
sex_surv = train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean() \
    .sort_values(by='Survived', ascending=False)
print(sex_surv)
sib_surv = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean() \
    .sort_values(by='Survived', ascending=False)
print(sib_surv)

parch_surv = train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean() \
    .sort_values(by='Survived', ascending=False)
print(parch_surv)
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
combine = [train, test]
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
train = train.drop(['AgeBand'], axis=1)
combine = [train, test]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                             ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                         ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gau = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_per = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_lin = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_forest = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree],
    'Prediction': [Y_pred_svc, Y_pred_knn, Y_pred_log,
                   Y_pred_forest, Y_pred_gau, Y_pred_per,
                   Y_pred_sgd, Y_pred_lin, Y_pred_tree]})

print(models.loc[:, ['Model', 'Score']].sort_values(by='Score', ascending=False))

max_score_idx = models['Score'].idxmax()
max_score = models['Score'].loc[max_score_idx]
max_score_pred = models['Prediction'].loc[max_score_idx]
max_score_model = models['Model'].loc[max_score_idx]

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": max_score_pred
})
with open('predictions_scores', mode='a') as file:
    file.write("Score: " + str(max_score) + "\t")
    file.write("Model: " + max_score_model + "\t")
    file.write("Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
submission.to_csv("submission.csv", mode="w", index=False)
# Attempt 1
# -------------------------------------------------------------------------------------
# test_ids = test["PassengerId"]
#
#
# def substrings_in_string(full_str: str, substrings: List[str]):
#     for substring in substrings:
#         if full_str.find(substring) != -1:
#             return substring
#     return np.nan
#
#
# def replace_titles(x):
#     title = x['Title']
#     if title in ['Lady', 'Countess','Capt', 'Col',
#                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
#         return 'Rare'
#     elif title == 'Mlle':
#         return 'Miss'
#     elif title == 'Ms':
#         return 'Miss'
#     elif title == "Mme":
#         return 'Mrs'
#     else:
#         return title
#
#
# def classify_by_age(x, median):
#     age = x['Age']
#     if age == np.nan:
#         age = median
#     if 0 <= age <= 14:
#         return 'child'
#     elif 15 <= age <= 25:
#         return 'not surviving prob'
#     elif 26 <= age <= 35:
#         return 'adult'
#     elif 36 <= age:
#         return 'elderly adult'
#     else:
#         raise ValueError
#
#
# def clean_data(data: pd.DataFrame) -> pd.DataFrame:
#     title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
#                   'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
#                   'Don', 'Jonkheer']
#     data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))
#     data['Title'] = data.apply(replace_titles, axis=1)
#     cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
#     data['Cabin'] = data['Cabin'].astype(str)
#     data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
#     data['Family_Size'] = data['SibSp'] + data['Parch']+1
#     data['Fare_per_Person'] = data['Fare'] / (data['Family_Size'] + 1)
#     data['Age_category'] = data.apply(classify_by_age, axis=1, median=data['Age'].median())
#     data['IsAlone'] = 0
#     data.loc[data['Family_Size']== 1,'IsAlone'] =1
#     data = data.drop(["Ticket", "PassengerId", "Cabin", "Name", "SibSp", "Parch", 'Family_Size'], axis=1)
#     cols = ["Fare", "Age", "Family_Size", "Fare_per_Person"]
#     for col in cols:
#         data[col].fillna(data[col].median(), inplace=True)
#     data.Embarked.fillna("U", inplace=True)
#     return data
#
#
# train = clean_data(train)
# test = clean_data(test)
#
# le = preprocessing.LabelEncoder()
# cols = ["Sex", "Embarked", "Title", "Deck", "Age_category"]
# for col in cols:
#     train[col] = le.fit_transform(train[col])
#     test[col] = le.transform(test[col])
#     print(le.classes_)
# print(train.head())
#
# y = train.pop("Survived")
# x = train
#
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
#
# clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)
# # clf = Ridge(alpha=1.0).fit(x_train, y_train)
# predictions = clf.predict(x_val)
# with open('predictions_scores', mode='a') as file:
#     acc_scr = accuracy_score(y_val, predictions)
#     print(accuracy_score(y_val, predictions))
#     file.write("Score: " + str(acc_scr) + f" {train.columns.values}\n")
#
# submission_preds = clf.predict(test)
# df = pd.DataFrame({"PassengerId": test_ids.values,
#                    "Survived": submission_preds,
#                    })
# df.to_csv("submission.csv", mode="w", index=False)
