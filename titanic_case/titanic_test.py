import os
import keras.optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
temp_dir = os.path.join(os.path.dirname(__file__), "../temp")
(gender_submission, test_path, train_path) = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir)]

train = pd.read_csv(train_path)
print(train.columns)
test = pd.read_csv(test_path)
print(test.columns)
gender_subm = pd.read_csv(gender_submission)
print(gender_subm.shape, gender_subm.columns)
CATEGORICAL_COLUMNS = ['Name', 'Sex',  'Ticket', 'Embarked']
NUMERICAL_COLUMNS = ['Pclass','Age', 'SibSp','Parch', 'Fare', 'Cabin']
train_label = train.pop('Survived')

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
def make_input_fn(data, label, epochs=10, shuffle=True, batch_size=128):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(data), label))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds
    return input_fn()


train_fn = make_input_fn(train, train_label)
# TODO: change gender_subm to custom or check linear classification
test_fn = make_input_fn(test, gender_subm, epochs=1, shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_fn)
result = linear_est.evaluate(test_fn)
print(result)

