"""Запустите этот файл кнопкой Run в PyCharm для создания submission.csv."""
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from features import make_features
from settings import CONFIG, THREADS


# Пути отсчитываются от main.py, поэтому рабочая папка PyCharm не важна.
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / 'dataset' / 'train.csv'
TEST_PATH = PROJECT_DIR / 'dataset' / 'test.csv'
OUTPUT_DIR = PROJECT_DIR / 'output'


def make_model(spec, seed, threads):
    return CatBoostClassifier(
        iterations=spec['iterations'], depth=spec['depth'],
        learning_rate=spec['learning_rate'], l2_leaf_reg=spec['l2_leaf_reg'],
        random_seed=seed, thread_count=threads, loss_function='Logloss',
        verbose=False, allow_writing_files=False, task_type='CPU',
    )


def choose_threshold(y, probability):
    """Сглаживание уменьшает зависимость от единичных пограничных объектов.

    Ищем не случайный пик F1, а устойчивый участок на сетке порогов.
    Порог всегда определяется по предсказаниям на отложенных строках.
    """
    grid = np.linspace(0.3, 0.75, 181)
    scores = np.array([f1_score(y, probability >= t, average='macro') for t in grid])
    smooth = np.convolve(scores, np.ones(9) / 9, mode='valid')
    best = np.flatnonzero(smooth >= smooth.max() - 0.0001) + 4
    return float(np.median(grid[best]))


def validate(train, config, threads):
    """Оценить ансамбль на отложенных строках и подобрать порог macro F1.

    По очереди обучаемся на четырех частях train и предсказываем пятую.
    Так каждая строка получает прогноз без обучения на ее target.
    Итоговые модели для test обучаются отдельно, на всем train.
    """
    y = train['target'].to_numpy()
    splitter = StratifiedKFold(n_splits=config['folds'], shuffle=True, random_state=config['seed'])
    splits = list(splitter.split(train, y))
    ensemble = np.zeros(len(train))
    total_weight = sum(spec['weight'] for spec in config['models'])

    for spec in config['models']:
        x = make_features(train, spec['features'])
        categories = list(x.select_dtypes(include=['object', 'str']).columns)
        probability = np.zeros(len(train))
        for fold, (fit_rows, valid_rows) in enumerate(splits):
            model = make_model(spec, config['seed'] + fold, threads)
            model.fit(x.iloc[fit_rows], y[fit_rows], cat_features=categories)
            probability[valid_rows] = model.predict_proba(x.iloc[valid_rows])[:, 1]
            print(f"Валидация {spec['name']}: {fold + 1}/{len(splits)}", flush=True)
        ensemble += spec['weight'] / total_weight * probability

    threshold = choose_threshold(y, ensemble)
    score = f1_score(y, ensemble >= threshold, average='macro')
    print(f'Валидация: macro F1 = {score:.6f}, порог = {threshold:.5f}', flush=True)
    return threshold


def predict(train, test, config, output, threads):
    """Усреднить вероятности моделей, обученных на ВСЕМ train с разными seed."""
    probability = np.zeros(len(test))
    total_weight = sum(spec['weight'] for spec in config['models'])
    for spec in config['models']:
        x = make_features(train, spec['features'])
        z = make_features(test, spec['features'])
        categories = list(x.select_dtypes(include=['object', 'str']).columns)
        for seed in config['final_seeds']:
            model = make_model(spec, seed, threads)
            model.fit(x, train['target'], cat_features=categories)
            weight = spec['weight'] / total_weight / len(config['final_seeds'])
            probability += weight * model.predict_proba(z)[:, 1]
            print(f"Обучена {spec['name']}, seed={seed}", flush=True)
    submission = pd.DataFrame({'id': test['id'].to_numpy(),
                               'target': (probability >= config['threshold']).astype(int)})
    submission.to_csv(output / 'submission.csv', index=False)
    print(f"Ответ сохранен: {output / 'submission.csv'}", flush=True)


def main():
    config = CONFIG.copy()
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config['threshold'] = validate(train, config, THREADS)
    predict(train, test, config, OUTPUT_DIR, THREADS)


if __name__ == '__main__':
    main()
