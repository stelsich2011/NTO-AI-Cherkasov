import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

cat_cols_init = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
for col in cat_cols_init:
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

def add_features(df, train_source):
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

    mean_temp_by_mat = train_source.groupby('material')['nozzle_temp_c'].mean().to_dict()
    global_mean = train_source['nozzle_temp_c'].mean()
    df['mat_mean_nozzle_temp'] = df['material'].map(mean_temp_by_mat).fillna(global_mean)
    df['material_nozzle_temp_diff'] = df['nozzle_temp_c'] - df['mat_mean_nozzle_temp']
    return df

X = add_features(X, train_source=X)
df_test = add_features(df_test, train_source=X)

X_test = df_test.drop(columns=['id'])

all_cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in all_cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

test_preds_cb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]

    cb_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=12,
        iterations=1500,
        random_state=42 + fold,
        loss_function='Logloss',
        eval_metric='F1',
        early_stopping_rounds=100,
        verbose=0
    )
    cb_model.fit(X_train, y_train, cat_features=all_cat_cols, eval_set=(X_val, y_val))
    
    oof_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_test)[:, 1] / n_splits

    lgb_model = LGBMClassifier(
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        random_state=42 + fold,
        objective='binary',
        metric='binary_logloss',
        importance_type='gain',
        verbose=-1
    )
    lgb_model.fit(
        X_train_lgb, y_train,
        eval_X=X_val_lgb,
        eval_y=y_val,
        callbacks=[] 
    )

    
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val_lgb)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / n_splits

print("\n Обучение завершено. Поиск оптимальных весов блендинга")

best_weight = 0.5
best_th = 0.5
best_f1 = 0

for weight in np.arange(0.3, 0.8, 0.05):
    oof_blend = weight * oof_cb + (1 - weight) * oof_lgb
    
    for th in np.arange(0.3, 0.7, 0.01):
        preds = (oof_blend >= th).astype(int)
        score = f1_score(y, preds, average='macro')
        if score > best_f1:
            best_f1 = score
            best_th = th
            best_weight = weight

print(f"Оптимальный вес CatBoost: {best_weight:.2f} (Вес LightGBM: {1-best_weight:.2f})")
print(f"Оптимальный порог отсечения: {best_th:.2f}")
print(f"Итоговый OOF macro F1 ансамбля: {best_f1:.4f}")

final_test_proba = best_weight * test_preds_cb + (1 - best_weight) * test_preds_lgb
final_predictions = (final_test_proba >= best_th).astype(int)

submission = pd.DataFrame({
    'id': df_test['id'], 
    'target': final_predictions
})
submission.to_csv('submission_ensemble_folds.csv', index=False)
print("Файл submission_ensemble_folds.csv сохранен")