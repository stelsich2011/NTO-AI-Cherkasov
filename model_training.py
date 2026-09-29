import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

X_test = df_test.drop(columns=['id'], errors='ignore')

cat_types = ['object', 'category', 'str', 'string']
cat_features = list(X.select_dtypes(include=cat_types).columns)

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg':[1, 3, 5],
    'iterations': [500, 1000]
}

model = CatBoostClassifier(cat_features=cat_features, random_state=42, verbose=False, early_stopping_rounds=50)

X[cat_features] = X[cat_features].fillna('missing').astype(str)
X_test[cat_features] = X_test[cat_features].fillna('missing').astype(str)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search_result = model.grid_search(
    param_grid, 
    X=X_train, 
    y=y_train, 
    cv=3,
    partition_random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

preds = model.predict(X_test)

submission = pd.DataFrame({
    'id': df_test['id'],
    'target': preds.flatten()
})

submission.to_csv('submission.csv', index=False)