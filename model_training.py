import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

model = CatBoostClassifier(iterations=1000, learning_rate=0.1, random_state=42)

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

X_test = df_test.drop(columns=['id'], errors='ignore')

cat_types = ['object', 'category', 'str', 'string']
cat_features = list(X.select_dtypes(include=cat_types).columns)

X[cat_features] = X[cat_features].fillna('missing').astype(str)
X_test[cat_features] = X_test[cat_features].fillna('missing').astype(str)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(
    X_train, 
    y_train, 
    cat_features=cat_features,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50, 
    verbose=100 
)

preds = model.predict(X_test)

submission = pd.DataFrame({
    'id': df_test['id'],
    'target': preds.flatten()
})

submission.to_csv('submission.csv', index=False)