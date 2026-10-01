import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer, roc_auc_score

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

cols_with_nan = ['bed_surface', 'filament_moisture_pct', 'ambient_temp_c', 'hours_since_maintenance']

num_cols = [c for c in cols_with_nan if pd.api.types.is_numeric_dtype(df_train[c])]
cat_cols = [c for c in cols_with_nan if c not in num_cols]

if num_cols:
    medians = df_train[num_cols].median()
    X[num_cols] = X[num_cols].fillna(medians)
    df_test[num_cols] = df_test[num_cols].fillna(medians)

if cat_cols:
    modes = df_train[cat_cols].mode().iloc[0]
    X[cat_cols] = X[cat_cols].fillna(modes)
    df_test[cat_cols] = df_test[cat_cols].fillna(modes)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

all_cat_cols = [
    col for col in X.columns 
    if not pd.api.types.is_numeric_dtype(X[col]) or col in cat_cols
]

for col in all_cat_cols:
    X[col] = X[col].astype(str)
    df_test[col] = df_test[col].astype(str)

macro_f1_scorer = make_scorer(f1_score, average='macro')

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.06, 0.1],
    'l2_leaf_reg': [5, 7, 10],
    'iterations': [600],
    'auto_class_weights': ['SqrtBalanced', 'Balanced']
}

base_model = CatBoostClassifier(
    verbose=0,
    random_state=42,
    loss_function='MultiClass'
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train, cat_features=all_cat_cols)

print("\nЛучшие параметры:", grid_search.best_params_)

best_params = grid_search.best_params_.copy()
best_params.pop('iterations', None)

final_model = CatBoostClassifier(
    **best_params,
    verbose=False,
    random_state=42,
    loss_function='MultiClass',
    early_stopping_rounds=75
)

final_model.fit(
    X_train, y_train,
    cat_features=all_cat_cols,
    eval_set=(X_val, y_val)
)

y_pred_val = final_model.predict(X_val)
val_f1 = f1_score(y_val, y_pred_val, average='macro')
print(f"\nmacro F1 на валидации: {val_f1:.4f}")

final_model_full = CatBoostClassifier(
    **best_params, 
    verbose=False, 
    random_state=42, 
    loss_function='MultiClass'
)
final_model_full.fit(X, y, cat_features=all_cat_cols)

X_test = df_test.drop(columns=['id'])
predictions = final_model_full.predict(X_test)

submission = pd.DataFrame({
    'id': df_test['id'], 
    'target': predictions.flatten() 
})
submission.to_csv('submission_catboost_binary_no.csv', index=False)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Топ-10 признаков по важности ---")
print(feature_importance.head(10).to_string(index=False))

