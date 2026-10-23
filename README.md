# Прогноз успешности 3D-печати

Проверено на Python 3.11.9. Основные зависимости:
NumPy, pandas, CatBoost и scikit-learn. Их версии указаны в `requirements.txt`;
SciPy и остальные вложенные зависимости pip установит автоматически.
GPU и отдельный компилятор не нужны.

1. Откройте папку проекта в PyCharm.
2. В **Settings → Project → Python Interpreter** выберите или создайте
   виртуальное окружение с Python 3.11.
3. В терминале PyCharm с активным окружением установите зависимости:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Положите `train.csv` и `test.csv` в папку `dataset` рядом с `main.py`.
5. Откройте `main.py` и нажмите **Run**. Параметры запуска не нужны.
6. Дождитесь завершения обучения. Ответ появится в `output/submission.csv`.

Настройки моделей находятся в `settings.py`.
