import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("ОБУЧЕНИЕ KNN С УЧЕТОМ СИЛАНА (исправлено)")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data_full.csv')
print(f"✅ Загружено {len(df)} записей")

# Заполняем пропуски (NaN) средними значениями для числовых колонок
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())
        print(f"  Заполнены пропуски в {col}")

# Кодируем категориальные признаки
encoders = {}
categorical_cols = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
for col in categorical_cols:
    if col in df.columns:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col].astype(str))

# Признаки
features = ['base_type_encoded', 'base_manufacturer_encoded', 'filler_type_encoded', 
            'filler_manufacturer_encoded', 'base_hardness', 'filler_content', 
            'silane_content', 'temp', 'time']
X = df[features]

# Целевые переменные (только те, где есть данные)
targets = ['strength_initial', 'elongation_initial', 'strength_aged_240h_250C',
           'elongation_aged_240h_250C', 'strength_aged_72h_250C', 
           'elongation_aged_72h_250C', 'resistivity', 'permittivity', 
           'tan_delta', 'dielectric_strength', 'ceramic_strength']

# Проверяем, что все целевые переменные есть в данных
targets = [t for t in targets if t in df.columns]
print(f"📊 Целевые переменные: {targets}")

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

for name, model in models.items():
    joblib.dump(model, models_dir / f'knn_{name}_{timestamp}.pkl')

joblib.dump(encoders, models_dir / f'knn_encoders_{timestamp}.pkl')
joblib.dump(scaler, models_dir / f'knn_scaler_{timestamp}.pkl')

print(f"\n✅ Модели сохранены (префикс: knn_{timestamp})")
print(f"   Использовано {len(df)} записей")

# Проверяем предсказание для шпинель с силаном
print("\n🔮 Проверка: Шпинель 40 phr + 5 phr силана")
test_data = pd.DataFrame([{
    'base_type': 'VMQ',
    'base_hardness': 70,
    'base_manufacturer': 'Xiameter',
    'filler_type': 'MgAl2O4',
    'filler_manufacturer': 'JSC_Vostochnye_Ogneupory',
    'filler_content': 40,
    'silane_content': 5,
    'temp': 115,
    'time': 15
}])

X_test = {}
for col, encoder in encoders.items():
    if col in test_data.columns:
        X_test[f'{col}_encoded'] = encoder.transform(test_data[col])[0]

X_test['base_hardness'] = test_data['base_hardness'].values[0]
X_test['filler_content'] = test_data['filler_content'].values[0]
X_test['silane_content'] = test_data['silane_content'].values[0]
X_test['temp'] = test_data['temp'].values[0]
X_test['time'] = test_data['time'].values[0]

X_df = pd.DataFrame([X_test])
X_scaled_test = scaler.transform(X_df)

pred = float(models['ceramic_strength'].predict(X_scaled_test)[0])
print(f"  ceramic_strength: {pred:.1f} Н/м²")
