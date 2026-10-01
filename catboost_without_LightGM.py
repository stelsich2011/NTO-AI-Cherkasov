import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

init_cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
for col in init_cat_cols:
    X[col] = X[col].astype(str)
    df_test[col] = df_test[col].astype(str)

cols_with_nan = ['bed_surface', 'filament_moisture_pct', 'ambient_temp_c', 'hours_since_maintenance']
num_cols = [c for c in cols_with_nan if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in cols_with_nan if c not in num_cols]

if num_cols:
    medians = X[num_cols].median()
    X[num_cols] = X[num_cols].fillna(medians)
    df_test[num_cols] = df_test[num_cols].fillna(medians)

if cat_cols:
    modes = X[cat_cols].mode().iloc[0]
    X[cat_cols] = X[cat_cols].fillna(modes)
    df_test[cat_cols] = df_test[cat_cols].fillna(modes)

def add_advanced_features(df, train_source):
    df['mat_printer'] = df['material'] + "_" + df['printer_model']
    df['mat_surface'] = df['material'] + "_" + df['bed_surface']

    df['temp_diff_nozzle_bed'] = df['nozzle_temp_c'] - df['bed_temp_c']
    df['temp_diff_bed_ambient'] = df['bed_temp_c'] - df['ambient_temp_c']

    df['mass_per_contact_area'] = df['model_mass_g'] / (df['contact_area_cm2'] + 1e-5)
    df['mass_per_contact_area_sq'] = df['mass_per_contact_area'] ** 2

    df['flow_rate_estimate'] = df['print_speed_mm_s'] * df['layer_height_mm'] * df['nozzle_diameter_mm'] * 1.2
    df['cooling_efficiency'] = df['cooling_percent'] / (df['print_speed_mm_s'] + 1e-5)
    df['temp_to_speed_ratio'] = df['nozzle_temp_c'] / (df['print_speed_mm_s'] + 1e-5)

    df['geometry_risk_score'] = df['geometry_complexity'] * df['overhang_angle_deg'] * df['estimated_time_min']
    df['maintenance_load'] = df['hours_since_maintenance'] * df['estimated_time_min']
    df['moisture_risk'] = df['filament_moisture_pct'] * df['ambient_temp_c']

    mean_temp_by_mat = train_source.groupby('material')['nozzle_temp_c'].mean().to_dict()
    global_mean_temp = train_source['nozzle_temp_c'].mean()
    df['mat_mean_nozzle_temp'] = df['material'].map(mean_temp_by_mat).fillna(global_mean_temp)
    df['material_nozzle_temp_diff'] = df['nozzle_temp_c'] - df['mat_mean_nozzle_temp']

    mean_speed_by_mat = train_source.groupby('material')['print_speed_mm_s'].mean().to_dict()
    global_mean_speed = train_source['print_speed_mm_s'].mean()
    df['mat_mean_speed'] = df['material'].map(mean_speed_by_mat).fillna(global_mean_speed)
    df['material_speed_diff'] = df['print_speed_mm_s'] - df['mat_mean_speed']

    return df

X = add_advanced_features(X, train_source=X)
df_test = add_advanced_features(df_test, train_source=X)

X_test = df_test.drop(columns=['id'])

all_cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
for col in all_cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("Старт обучения CatBoost на новых признаках")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=12,
        iterations=2000,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_state=42 + fold,
        loss_function='Logloss',
        eval_metric='F1',
        early_stopping_rounds=150,
        verbose=0
    )
    
    model.fit(X_train, y_train, cat_features=all_cat_cols, eval_set=(X_val, y_val))

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    print(f"Фолд {fold+1} готов. Best Iteration: {model.get_best_iteration()}")

best_th = 0.5
best_f1 = 0
for th in np.arange(0.2, 0.8, 0.01):
    preds = (oof_preds >= th).astype(int)
    score = f1_score(y, preds, average='macro')
    if score > best_f1:
        best_f1 = score
        best_th = th

print(f"\n Результаты")
print(f"Оптимальный порог отсечения: {best_th:.2f}")
print(f"Итоговый OOF macro F1: {best_f1:.5f}")

final_predictions = (test_preds >= best_th).astype(int)

submission = pd.DataFrame({
    'id': df_test['id'], 
    'target': final_predictions
})
submission.to_csv('submission_catboost_advanced_fe.csv', index=False)
print("Файл submission_catboost_advanced_fe.csv сохранен")
