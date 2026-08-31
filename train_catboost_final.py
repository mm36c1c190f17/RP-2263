import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from pathlib import Path

print("=" * 60)
print("ОБУЧЕНИЕ CATBOOST ДЛЯ ПРЕДСКАЗАНИЯ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")

# Показываем данные по CaSiO3
print("\n📊 Данные для CaSiO3:")
print(df[df['filler_type'] == 'CaSiO3'][['filler_content', 'silane_content', 'ceramic_strength']])

# Определяем колонки
cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer', 'silane_type']
num_features = ['base_hardness', 'filler_content', 'silane_content', 'temp', 'time']
feature_names = cat_features + num_features

X = df[feature_names]
y = df['ceramic_strength']

# Заполняем пропуски
for col in num_features:
    X[col] = X[col].fillna(X[col].mean())

# Разделяем
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Train: {len(X_train)} записей, Test: {len(X_test)} записей")

# Обучаем CatBoost с параметрами для малых данных
model = CatBoostRegressor(
    iterations=500,          # больше итераций
    learning_rate=0.05,      # меньше шаг
    depth=6,                 # глубже дерево
    l2_leaf_reg=3,          # регуляризация
    random_seed=42,
    verbose=100,
    loss_function='RMSE'
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    verbose=100
)

# Оценка
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\n📊 Результаты:")
print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
print(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# Сохраняем
models_dir = Path('models/')
models_dir.mkdir(exist_ok=True)

model.save_model('models/catboost_predictor.cbm')
joblib.dump(feature_names, 'models/feature_names.pkl')
joblib.dump(cat_features, 'models/cat_features.pkl')

print("\n✅ Модель сохранена")

# Проверка на известных составах
print("\n🔮 Проверка предсказаний:")
test_cases = [
    ('CaSiO3', 30, 0, 98.8),
    ('CaSiO3', 40, 5, 250.0),
    ('CaSiO3', 30, 5, None),  # Этого нет в данных
]

for filler, content, silane, real in test_cases:
    input_data = pd.DataFrame([{
        'base_type': 'VMQ',
        'base_manufacturer': 'Xiameter',
        'filler_type': filler,
        'filler_manufacturer': 'JSC_Geokom',
        'silane_type': 'A-1120' if silane > 0 else '0',
        'base_hardness': 70,
        'filler_content': content,
        'silane_content': silane,
        'temp': 115,
        'time': 15
    }])
    
    input_data = input_data[feature_names]
    pred = model.predict(input_data)[0]
    
    if real is not None:
        print(f"  {filler} {content} phr + {silane} phr: реальное={real:.1f}, предсказанное={pred:.1f}")
    else:
        print(f"  {filler} {content} phr + {silane} phr: предсказанное={pred:.1f} (НОВЫЙ СОСТАВ)")

print("\n✅ Модель готова к использованию!")
