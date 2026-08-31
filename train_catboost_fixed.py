import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ОБУЧЕНИЕ CATBOOST НА 9 ЗАПИСЯХ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")

# Показываем данные
print("\n📊 Данные для обучения:")
print(df[['filler_type', 'filler_content', 'ceramic_strength']])

# Категориальные признаки
cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
num_features = ['base_hardness', 'filler_content', 'temp', 'time']

X = df[cat_features + num_features]

# Целевая переменная - ceramic_strength
y = df['ceramic_strength']

print(f"\n🔬 Обучаем модель для ceramic_strength...")
print(f"   Категориальные признаки: {cat_features}")
print(f"   Числовые признаки: {num_features}")

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=4,
    verbose=50,
    random_seed=42,
    loss_function='RMSE'
)

model.fit(X, y, cat_features=cat_features, verbose=50)

print(f"\n✅ Модель обучена на {len(df)} записях")
model.save_model('models/catboost_ceramic_strength.cbm')

# Проверяем предсказания
print("\n📊 Проверка на обучающих данных:")
for filler in df['filler_type'].unique():
    test_data = df[df['filler_type'] == filler].iloc[0:1]
    X_test = test_data[cat_features + num_features]
    pred = model.predict(X_test)[0]
    real = test_data['ceramic_strength'].values[0]
    print(f"  {filler}: реальное={real:.1f}, предсказанное={pred:.1f} (ошибка: {abs(pred-real):.1f})")

print("\n✅ Модель сохранена!")
