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

# Кодируем категориальные признаки
cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer', 'silane_type']
num_features = ['base_hardness', 'filler_content', 'silane_content', 'temp', 'time']

X = df[cat_features + num_features]
y = df['ceramic_strength']

# Заполняем пропуски в числовых признаках
for col in num_features:
    X[col] = X[col].fillna(X[col].mean())

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Train: {len(X_train)} записей, Test: {len(X_test)} записей")

# Обучаем CatBoost с параметрами для малых данных
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.3,
    depth=4,
    l2_leaf_reg=3,
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

# Сохраняем модель
models_dir = Path('models/')
models_dir.mkdir(exist_ok=True)

model.save_model('models/catboost_predictor.cbm')
joblib.dump(cat_features, 'models/catboost_features.pkl')

print("\n✅ Модель сохранена как catboost_predictor.cbm")

# Проверка предсказаний для известных составов
print("\n🔮 Проверка на известных составах:")
test_cases = [
    ('MgAl2O4', 30, 0, 38.3),
    ('CaSiO3', 30, 0, 98.8),
    ('Al2O3', 30, 0, 96.6),
    ('SiO2', 30, 0, 71.6),
]

for filler, content, silane, real in test_cases:
    test_data = pd.DataFrame([{
        'base_type': 'VMQ',
        'base_hardness': 70,
        'base_manufacturer': 'Xiameter',
        'filler_type': filler,
        'filler_manufacturer': 'JSC_Vostochnye_Ogneupory',
        'silane_type': '0',
        'silane_content': silane,
        'temp': 115,
        'time': 15
    }])
    
    pred = model.predict(test_data)[0]
    print(f"  {filler} {content} phr + {silane} phr: реальное={real:.1f}, предсказанное={pred:.1f}")
