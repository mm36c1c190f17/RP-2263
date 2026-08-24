import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
print("=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ ВСЕХ СВОЙСТВ")
print("=" * 60)

# Пробуем загрузить реальные данные
try:
    data = pd.read_excel('data/rubber_data.xlsx')
    print(f"\n✅ Загружены РЕАЛЬНЫЕ данные из rubber_data.xlsx")
    print(f"   Строк: {len(data)}")
    print(f"   Столбцы: {list(data.columns)}")
except FileNotFoundError:
    print("\n⚠️ Файл rubber_data.xlsx не найден, используем синтетические данные")
    data = pd.read_csv('data/synthetic_data.csv')
    print(f"   Загружено {len(data)} строк")
    # Переименовываем столбцы для совместимости
    data = data.rename(columns={
        'ingredient_a': 'component_a',
        'ingredient_b': 'component_b',
        'ingredient_c': 'component_c',
        'temperature': 'temp',
        'time': 'time',
        'strength': 'strength_initial',
        'elongation': 'elongation_initial'
    })

# ============================================================
# 2. ОПРЕДЕЛЯЕМ ВХОДНЫЕ ПРИЗНАКИ
# ============================================================
# Основные признаки (ингредиенты + режим)
features = ['component_a', 'component_b', 'component_c', 'temp', 'time']

# Проверяем, есть ли столбцы с наполнителями из твоей дипломной
optional_features = ['spinel', 'wollastonite', 'alumina', 'cao', 'silica']
available_features = []

for f in optional_features:
    if f in data.columns:
        available_features.append(f)

# Добавляем найденные наполнители к признакам
all_features = features + available_features
print(f"\n📊 Используемые признаки: {all_features}")

# ============================================================
# 3. ОПРЕДЕЛЯЕМ ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ (ЧТО БУДЕМ ПРЕДСКАЗЫВАТЬ)
# ============================================================
targets = {
    'strength_initial': 'Прочность до старения, МПа',
    'elongation_initial': 'Удлинение до старения, %',
    'strength_aged_24h': 'Прочность после 24ч, МПа',
    'elongation_aged_24h': 'Удлинение после 24ч, %',
    'strength_aged_72h': 'Прочность после 72ч, МПа',
    'elongation_aged_72h': 'Удлинение после 72ч, %',
    'resistivity': 'Удельное сопротивление, Ом·м',
    'permittivity': 'Диэлектрическая проницаемость',
    'tan_delta': 'Тангенс угла потерь',
    'dielectric_strength': 'Электрическая прочность, МВ/м',
    'ceramic_strength': 'Прочность керамического остатка, Н/м²'
}

# Проверяем, какие из целевых переменных есть в данных
available_targets = {}
for key, name in targets.items():
    if key in data.columns:
        available_targets[key] = name

print(f"\n🎯 Целевые переменные для обучения ({len(available_targets)} шт.):")
for key, name in available_targets.items():
    print(f"   - {key}: {name}")

if len(available_targets) == 0:
    print("\n❌ Нет целевых переменных! Проверь названия столбцов.")
    print(f"   Доступные столбцы: {list(data.columns)}")
    exit()

# ============================================================
# 4. РАЗДЕЛЯЕМ ДАННЫЕ
# ============================================================
X = data[all_features]

# ============================================================
# 5. ОБУЧАЕМ МОДЕЛИ
# ============================================================
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

os.makedirs('models', exist_ok=True)

results = {}

for target_key, target_name in available_targets.items():
    print(f"\n📊 Обучаем модель для: {target_name}")
    
    # Проверяем, есть ли целевая переменная
    if target_key not in data.columns:
        print(f"   ⚠️ Пропускаем: столбец {target_key} не найден")
        continue
    
    y = data[target_key]
    
    # Удаляем строки с пропущенными значениями
    mask = y.notna()
    if not mask.all():
        print(f"   ⚠️ Найдены пропуски, используем {mask.sum()} строк из {len(y)}")
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 5:
        print(f"   ❌ Слишком мало данных ({len(X_clean)} строк), пропускаем")
        continue
    
    # Делим на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # Обучаем модель
    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        verbose=50,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    
    # Оценка качества
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Сохраняем модель
    model_path = f'models/catboost_{target_key}.cbm'
    model.save_model(model_path)
    
    results[target_key] = {
        'name': target_name,
        'mae': mae,
        'r2': r2,
        'samples': len(X_clean),
        'path': model_path
    }
    
    print(f"   ✅ Модель сохранена: {model_path}")
    print(f"   📊 MAE: {mae:.3f}, R²: {r2:.3f}")

# ============================================================
# 6. ИТОГИ
# ============================================================
print("\n" + "=" * 60)
print("ИТОГИ ОБУЧЕНИЯ")
print("=" * 60)

print(f"\n✅ Успешно обучено {len(results)} моделей:\n")

for key, res in results.items():
    print(f"   📁 {key}")
    print(f"      {res['name']}")
    print(f"      MAE: {res['mae']:.3f}, R²: {res['r2']:.3f}, Данных: {res['samples']} строк")
    print(f"      Файл: {res['path']}")

# Проверяем, есть ли модели для сайта
site_models = ['strength_initial', 'elongation_initial', 'ceramic_strength']
print("\n" + "=" * 60)
print("ГОТОВНОСТЬ К ИСПОЛЬЗОВАНИЮ НА САЙТЕ")
print("=" * 60)

for model_key in site_models:
    if model_key in results:
        print(f"   ✅ {model_key} - готова")
    else:
        print(f"   ⚠️ {model_key} - не обучена (проверь данные)")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)