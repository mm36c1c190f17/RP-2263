import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загружаем данные
print("Загружаем данные...")
data = pd.read_csv('data/synthetic_data.csv')
print(f"Загружено {len(data)} строк")

# 2. Разделяем на признаки (X) и целевую переменную (y)
# Предположим, мы хотим предсказывать прочность (strength)
X = data[['ingredient_a', 'ingredient_b', 'ingredient_c', 'temperature', 'time']]
y = data['strength']

# 3. Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Обучающая выборка: {len(X_train)} строк")
print(f"Тестовая выборка: {len(X_test)} строк")

# 4. Создаём и обучаем модель CatBoost
print("Обучаем модель CatBoost...")
model = CatBoostRegressor(
    iterations=100,           # количество деревьев
    learning_rate=0.1,        # скорость обучения
    depth=4,                  # глубина деревьев
    verbose=20,               # показывать прогресс каждые 20 итераций
    random_seed=42
)

model.fit(X_train, y_train)

# 5. Делаем предсказания на тестовой выборке
y_pred = model.predict(X_test)

# 6. Оцениваем точность
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Результаты ===")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.3f}")

# 7. Показываем пример предсказания
print("\nПримеры предсказаний (первые 3 из тестовой выборки):")
for i in range(3):
    print(f"Фактическое: {y_test.iloc[i]:.1f}, Предсказанное: {y_pred[i]:.1f}")

# 8. Сохраняем модель
import os
os.makedirs('models', exist_ok=True)
model.save_model('models/catboost_strength.cbm')
print("\nМодель сохранена в папку models/catboost_strength.cbm")