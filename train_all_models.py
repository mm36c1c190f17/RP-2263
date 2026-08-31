import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

print("=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ НА РАСШИРЕННЫХ ДАННЫХ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
data = None

# Пробуем загрузить расширенный файл
try:
    data = pd.read_excel('data/rubber_data_extended.xlsx')
    print("\n✅ Загружены РАСШИРЕННЫЕ данные из rubber_data_extended.xlsx")
    print(f"   Строк: {len(data)}")
    print(f"   Столбцы: {list(data.columns)}")
except FileNotFoundError:
    print("\n⚠️ Файл rubber_data_extended.xlsx не найден, пробуем старый формат...")
    try:
        data = pd.read_excel('data/rubber_data.xlsx')
        print("\n✅ Загружены старые данные из rubber_data.xlsx")
        print(f"   Строк: {len(data)}")
        print(f"   Столбцы: {list(data.columns)}")
    except FileNotFoundError:
        print("\n⚠️ Используем синтетические данные")
        data = pd.read_csv('data/synthetic_data.csv')
        print(f"   Загружено {len(data)} строк")
        # Переименовываем для совместимости
        data = data.rename(columns={
            'ingredient_a': 'component_a',
            'ingredient_b': 'component_b',
            'ingredient_c': 'component_c',
            'temperature': 'temp',
            'time': 'time',
            'strength': 'strength_initial',
            'elongation': 'elongation_initial'
        })

if data is None:
    print("\n❌ Не удалось загрузить данные!")
    exit()

# ============================================================
# 2. ОПРЕДЕЛЯЕМ ПРИЗНАКИ (ВХОДНЫЕ ДАННЫЕ)
# ============================================================
# Базовые признаки
base_features = ['temp', 'time']

# Признаки для расширенной таблицы
extended_features = [
    'base_type', 'base_hardness', 'base_manufacturer',
    'filler_type', 'filler_content', 'filler_manufacturer'
]

# Признаки для старой таблицы (синтетика или rubber_data.xlsx)
old_features = ['component_a', 'component_b', 'component_c']

# Определяем, какие признаки есть в данных
features = []
for f in base_features:
    if f in data.columns:
        features.append(f)

# Проверяем расширенные признаки
has_extended = all(f in data.columns for f in extended_features)
if has_extended:
    features.extend(extended_features)
    print("\n📊 Используем РАСШИРЕННЫЕ признаки (с производителями)")
else:
    # Проверяем старые признаки
    has_old = all(f in data.columns for f in old_features)
    if has_old:
        features.extend(old_features)
        print("\n📊 Используем СТАНДАРТНЫЕ признаки (component_a, b, c)")
    else:
        print("\n❌ Не удалось определить признаки для обучения!")
        print(f"   Доступные столбцы: {list(data.columns)}")
        exit()

print(f"   Признаки: {features}")

# ============================================================
# 3. ОПРЕДЕЛЯЕМ ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ
# ============================================================
targets = {
    'strength_initial': 'Прочность до старения, МПа',
    'elongation_initial': 'Удлинение до старения, %',
    'strength_aged_240h_250C': 'Прочность после 240ч при 250°C, МПа',
    'elongation_aged_240h_250C': 'Удлинение после 240ч при 250°C, %',
    'strength_aged_72h_250C': 'Прочность после 72ч при 250°C, МПа',
    'elongation_aged_72h_250C': 'Удлинение после 72ч при 250°C, %',
    'resistivity': 'Удельное сопротивление, Ом·м',
    'permittivity': 'Диэлектрическая проницаемость',
    'tan_delta': 'Тангенс угла потерь',
    'dielectric_strength': 'Электрическая прочность, МВ/м',
    'ceramic_strength': 'Прочность керамического остатка, Н/м²'
}

# Проверяем, какие целевые переменные есть в данных
available_targets = {}
for key, name in targets.items():
    if key in data.columns:
        available_targets[key] = name

print(f"\n🎯 Целевые переменные ({len(available_targets)} шт.):")
for key, name in available_targets.items():
    print(f"   - {key}: {name}")

if len(available_targets) == 0:
    print("\n❌ Нет целевых переменных! Проверь названия столбцов.")
    exit()

X = data[features]

# ============================================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

os.makedirs('models', exist_ok=True)

# Определяем категориальные признаки (если они есть)
categorical_features = []
for f in ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']:
    if f in features:
        categorical_features.append(f)

results = {}

for target_key, target_name in available_targets.items():
    print(f"\n📊 Обучаем модель для: {target_name}")
    
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
        random_seed=42,
        cat_features=categorical_features if categorical_features else None
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
# 5. ИТОГИ
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

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)