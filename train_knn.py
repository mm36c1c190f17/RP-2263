import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("ОБУЧЕНИЕ МОДЕЛИ KNN НА ПРАВИЛЬНЫХ ДАННЫХ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")
print(f"📊 Наполнители: {df['filler_type'].unique().tolist()}")

# Кодируем категориальные признаки
encoders = {}
categorical_cols = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
for col in categorical_cols:
    if col in df.columns:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col].astype(str))

# Признаки
features = ['base_type_encoded', 'base_manufacturer_encoded', 'filler_type_encoded', 
            'filler_manufacturer_encoded', 'base_hardness', 'filler_content', 'temp', 'time']
X = df[features]

# Целевые переменные
targets = ['strength_initial', 'elongation_initial', 'strength_aged_240h_250C',
           'elongation_aged_240h_250C', 'strength_aged_72h_250C', 
           'elongation_aged_72h_250C', 'resistivity', 'permittivity', 
           'tan_delta', 'dielectric_strength', 'ceramic_strength']
y = df[targets]

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучаем KNN
models = {}
for target in targets:
    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(X_scaled, y[target])
    models[target] = model
    print(f"✅ {target} - обучено")

# Сохраняем
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
models_dir = Path('models/')
models_dir.mkdir(exist_ok=True)

print(f"\n💾 Сохраняем модели...")
for name, model in models.items():
    joblib.dump(model, models_dir / f'{name}_{timestamp}.pkl')

joblib.dump(encoders, models_dir / f'encoders_{timestamp}.pkl')
joblib.dump(scaler, models_dir / f'scaler_{timestamp}.pkl')

print(f"\n✅ Модели сохранены в models/ (префикс: {timestamp})")
print(f"   Использовано {len(df)} записей")
