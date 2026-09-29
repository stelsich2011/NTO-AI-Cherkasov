Python версии 3.11.9
numpy == 1.26.4
pandas == 2.2.2
scikit-learn == 1.4.2
catboost == 1.2.5

ИНСТРУКЦИЯ ПО ЗАПУСКУ:
В рабочей директории должна быть папка 'dataset' с файлами 
'train.csv' и 'test.csv'
Установите необходимые библиотеки:
'pip install numpy pandas scikit-learn catboost'
Запустите скрипт из терминала без дополнительных аргументов:
'python solution.py'
По окончании работы скрипта в корневой директории автоматически сгенерируется 
файл 'submission.csv', содержащий ровно две колонки: 'id' и 'target'.