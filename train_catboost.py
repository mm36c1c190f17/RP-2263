import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

print("=" * 60)
print("ОБУЧЕНИЕ CATBOOST НА ПРАВИЛЬНЫХ ДАННЫХ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")
print(f"📊 Наполнители: {df['filler_type'].unique().tolist()}")

# Определяем категориальные и числовые признаки
cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
num_features = ['base_hardness', 'filler_content', 'temp', 'time']

# Целевые переменные
targets = ['strength_initial', 'elongation_initial', 'strength_aged_240h_250C',
           'elongation_aged_240h_250C', 'strength_aged_72h_250C', 
           'elongation_aged_72h_250C', 'resistivity', 'permittivity', 
           'tan_delta', 'dielectric_strength', 'ceramic_strength']

X = df[cat_features + num_features]

# Обучаем модели для каждой целевой переменной
for target in targets:
    print(f"\n🔬 Обучение для {target}...")
    y = df[target]
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_seed=42
    )
    
    model.fit(X, y, cat_features=cat_features)
    model.save_model(f'models/catboost_{target}.cbm')
    print(f"  ✅ {target} - сохранено")

print("\n✅ Все модели CatBoost переобучены на правильных данных!")
