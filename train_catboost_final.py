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

# Определяем колонки
cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer', 'silane_type']
num_features = ['base_hardness', 'filler_content', 'silane_content', 'temp', 'time']

# Все колонки в правильном порядке
feature_names = cat_features + num_features

X = df[feature_names]
y = df['ceramic_strength']

# Заполняем пропуски
for col in num_features:
    X[col] = X[col].fillna(X[col].mean())

# Разделяем
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Train: {len(X_train)} записей, Test: {len(X_test)} записей")

# Обучаем
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=50,
    loss_function='RMSE'
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    verbose=50
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
print(f"   Порядок колонок: {feature_names}")
