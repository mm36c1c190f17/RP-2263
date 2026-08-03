# config.py - настройки проекта

import os

# Базовые пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути к данным
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Пути к моделям
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Имя файла с рецептурами (тот, что ты только что переместил)
RECIPES_FILE = os.path.join(RAW_DATA_DIR, 'D-11;14 Подлесский RPA 173-60.xlsx')

# Имя файла с обработанными данными
PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, 'rubber_data.csv')

# Убеждаемся, что папки существуют
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)