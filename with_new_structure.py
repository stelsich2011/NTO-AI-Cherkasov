import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

all_cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

for col in all_cat_cols:
    X[col] = X[col].astype(str)
    df_test[col] = df_test[col].astype(str)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cols_with_nan = ['bed_surface', 'filament_moisture_pct', 'ambient_temp_c', 'hours_since_maintenance']
num_cols = [c for c in cols_with_nan if pd.api.types.is_numeric_dtype(X_train[c])]
cat_cols = [c for c in cols_with_nan if c not in num_cols]

if num_cols:
    medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_val[num_cols] = X_val[num_cols].fillna(medians)
    X[num_cols] = X[num_cols].fillna(medians)
    df_test[num_cols] = df_test[num_cols].fillna(medians)

if cat_cols:
    modes = X_train[cat_cols].mode().iloc[0]
    X_train[cat_cols] = X_train[cat_cols].fillna(modes)
    X_val[cat_cols] = X_val[cat_cols].fillna(modes)
    X[cat_cols] = X[cat_cols].fillna(modes)
    df_test[cat_cols] = df_test[cat_cols].fillna(modes)

def add_features(df, train_source=None):
    df['temp_diff_nozzle_bed'] = df['nozzle_temp_c'] - df['bed_temp_c']
    df['temp_diff_bed_ambient'] = df['bed_temp_c'] - df['ambient_temp_c']
    df['mass_per_contact_area'] = df['model_mass_g'] / (df['contact_area_cm2'] + 1e-5)
    df['flow_rate_estimate'] = df['print_speed_mm_s'] * df['layer_height_mm'] * df['nozzle_diameter_mm'] * 1.2
    df['geometry_risk_score'] = df['geometry_complexity'] * df['overhang_angle_deg'] * df['estimated_time_min']
    df['maintenance_load'] = df['hours_since_maintenance'] * df['estimated_time_min']
    df['temp_to_speed_ratio'] = df['nozzle_temp_c'] / (df['print_speed_mm_s'] + 1e-5)
    df['time_per_gram'] = df['estimated_time_min'] / (df['model_mass_g'] + 1e-5)
    df['layers_count_estimate'] = df['model_mass_g'] / (df['layer_height_mm'] + 1e-5)
    df['moisture_risk'] = df['filament_moisture_pct'] * df['ambient_temp_c']
    df['cooling_efficiency'] = df['cooling_percent'] / (df['print_speed_mm_s'] + 1e-5)

    if train_source is not None:
        mean_temp_by_mat = train_source.groupby('material')['nozzle_temp_c'].mean().to_dict()
        global_mean = train_source['nozzle_temp_c'].mean()
        df['mat_mean_nozzle_temp'] = df['material'].map(mean_temp_by_mat).fillna(global_mean)
    else:
        mean_temp_by_mat = df.groupby('material')['nozzle_temp_c'].mean().to_dict()
        df['mat_mean_nozzle_temp'] = df['material'].map(mean_temp_by_mat)
        
    df['material_nozzle_temp_diff'] = df['nozzle_temp_c'] - df['mat_mean_nozzle_temp']
    return df

X_train = add_features(X_train, train_source=X_train)
X_val = add_features(X_val, train_source=X_train)
X = add_features(X, train_source=X)
df_test = add_features(df_test, train_source=X)

macro_f1_scorer = make_scorer(f1_score, average='macro')

param_grid = {
    'depth': [6, 8, 10],
    'learning_rate': [0.03, 0.06, 0.1],
    'l2_leaf_reg': [3, 7, 12],
    'iterations': [700]
}

base_model = CatBoostClassifier(
    verbose=0,
    random_state=42,
    loss_function='Logloss',
    eval_metric='F1'
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
    iterations=1500,
    verbose=100,
    random_state=42,
    loss_function='Logloss',
    eval_metric='F1',
    early_stopping_rounds=100
)

final_model.fit(
    X_train, y_train,
    cat_features=all_cat_cols,
    eval_set=(X_val, y_val)
)

y_proba_val = final_model.predict_proba(X_val)[:, 1]

best_th = 0.5
best_f1 = 0

for th in np.arange(0.1, 0.9, 0.01):
    y_pred_th = (y_proba_val >= th).astype(int)
    score = f1_score(y_val, y_pred_th, average='macro')
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"\nОптимальный порог: {best_th:.2f} -> Лучший macro F1 на валидации: {best_f1:.4f}")

iterations_optimal = final_model.get_best_iteration() or 800

final_model_full = CatBoostClassifier(
    **best_params, 
    iterations=int(iterations_optimal * 1.1), 
    verbose=100, 
    random_state=42, 
    loss_function='Logloss'
)
final_model_full.fit(X, y, cat_features=all_cat_cols)

X_test = df_test.drop(columns=['id'])
test_probabilities = final_model_full.predict_proba(X_test)[:, 1]
final_predictions = (test_probabilities >= best_th).astype(int)

submission = pd.DataFrame({
    'id': df_test['id'], 
    'target': final_predictions
})
submission.to_csv('submission_catboost_binary_fix.csv', index=False)
print("Файл submission_catboost_binary_fix.csv успешно сохранен.")
