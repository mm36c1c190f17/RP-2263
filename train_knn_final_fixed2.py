import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("ОБУЧЕНИЕ KNN С ТОЧНОЙ ИДЕНТИФИКАЦИЕЙ НАПОЛНИТЕЛЕЙ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')
print(f"✅ Загружено {len(df)} записей")

# Кодируем ВСЕ категориальные признаки
encoders = {}
categorical_cols = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
for col in categorical_cols:
    if col in df.columns:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col].astype(str))

# ВАЖНО: добавляем ВСЕ признаки, включая filler_type и filler_manufacturer
features = ['base_type_encoded', 'base_manufacturer_encoded', 'filler_type_encoded', 
            'filler_manufacturer_encoded', 'base_hardness', 'filler_content', 
            'silane_content', 'temp', 'time']

X = df[features]

# Целевые переменные
targets = ['ceramic_strength', 'strength_initial', 'elongation_initial', 
           'strength_aged_240h_250C', 'elongation_aged_240h_250C',
           'strength_aged_72h_250C', 'elongation_aged_72h_250C',
           'resistivity', 'permittivity', 'tan_delta', 'dielectric_strength']

# Заполняем пропуски
for col in targets:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

y = df[targets]

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучаем KNN с разными значениями k
for k in [1, 2, 3]:
    print(f"\n🔬 Обучаем KNN с k={k}")
    models = {}
    for target in targets:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_scaled, y[target])
        models[target] = model
    
    # Проверяем
    print(f"\n📊 Проверка для k={k}:")
    test_cases = [
        ('CaO', 30, 0, 30.0),
        ('MgAl2O4', 30, 0, 38.3),
        ('CaSiO3', 30, 0, 98.8),
        ('Al2O3', 30, 0, 96.6),
        ('SiO2', 30, 0, 71.6),
    ]
    
    errors = []
    for filler, content, silane, real in test_cases:
        data = {
            'base_type': 'VMQ',
            'base_hardness': 70,
            'base_manufacturer': 'Xiameter',
            'filler_type': filler,
            'filler_manufacturer': 'JSC_Vostochnye_Ogneupory',
            'filler_content': content,
            'silane_content': silane,
            'temp': 115,
            'time': 15
        }
        
        X_dict = {}
        for col, encoder in encoders.items():
            if col in data:
                try:
                    X_dict[f'{col}_encoded'] = encoder.transform([data[col]])[0]
                except:
                    X_dict[f'{col}_encoded'] = 0
        
        X_dict['base_hardness'] = data['base_hardness']
        X_dict['filler_content'] = data['filler_content']
        X_dict['silane_content'] = data['silane_content']
        X_dict['temp'] = data['temp']
        X_dict['time'] = data['time']
        
        X_df = pd.DataFrame([X_dict])
        X_scaled_test = scaler.transform(X_df)
        
        pred = float(models['ceramic_strength'].predict(X_scaled_test)[0])
        error = abs(pred - real)
        errors.append(error)
        status = '✅' if error < 0.5 else '❌'
        print(f'{status} {filler} {content} phr: реальное={real:.1f}, предсказанное={pred:.1f}')
    
    if max(errors) < 0.5:
        print(f"\n✅ k={k} работает правильно!")
        # Сохраняем эту модель
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path('models/')
        models_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            joblib.dump(model, models_dir / f'knn_{name}_{timestamp}.pkl')
        
        joblib.dump(encoders, models_dir / f'knn_encoders_{timestamp}.pkl')
        joblib.dump(scaler, models_dir / f'knn_scaler_{timestamp}.pkl')
        
        print(f"\n✅ Модель с k={k} сохранена (префикс: knn_{timestamp})")
        break
    else:
        print(f"❌ k={k} не работает, пробуем другое значение...")
