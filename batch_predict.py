import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def load_models():
    models_dir = Path("models/")
    encoders_files = list(models_dir.glob("encoders_*.pkl"))
    if not encoders_files:
        raise FileNotFoundError("Модели не найдены!")
    
    latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace("encoders_", "")
    
    encoders = joblib.load(f"models/encoders_{timestamp}.pkl")
    scaler = joblib.load(f"models/scaler_{timestamp}.pkl")
    
    models = {}
    for f in models_dir.glob(f"*_{timestamp}.pkl"):
        name = f.stem.replace(f"_{timestamp}", "")
        if name not in ['encoders', 'scaler', 'feature_info']:
            models[name] = joblib.load(f)
    
    return models, encoders, scaler

def predict_composition(filler_type, silane=0, amount=40):
    models, encoders, scaler = load_models()
    
    data = {
        'filler_type': filler_type,
        'manufacturer': 'Дубна',
        'rubber_type': 'СКТВ-1',
        'silane_content': silane,
        'filler_amount': amount
    }
    
    X_dict = {}
    for col, encoder in encoders.items():
        if col in data:
            try:
                X_dict[f'{col}_encoded'] = encoder.transform([data[col]])[0]
            except:
                X_dict[f'{col}_encoded'] = encoder.transform([encoder.classes_[0]])[0]
    
    X_dict['silane_content'] = data['silane_content']
    X_dict['filler_amount'] = data['filler_amount']
    
    X_df = pd.DataFrame([X_dict])
    X_scaled = scaler.transform(X_df)
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_scaled)[0]
    
    return predictions

# Создаем таблицу со всеми комбинациями
fillers = ['шпинель', 'волластонит', 'Al2O3']
silane_options = [0, 5]
amounts = [30, 40, 50]

results = []
for filler in fillers:
    for silane in silane_options:
        for amount in amounts:
            try:
                pred = predict_composition(filler, silane, amount)
                results.append({
                    'filler': filler,
                    'silane': silane,
                    'amount': amount,
                    **pred
                })
                print(f"✅ {filler} {silane}phr {amount}phr")
            except Exception as e:
                print(f"❌ {filler} {silane}phr {amount}phr: {e}")

# Сохраняем результаты
df = pd.DataFrame(results)
df.to_csv('all_predictions.csv', index=False)
print(f"\n✅ Результаты сохранены в all_predictions.csv")
print(f"📊 Всего предсказаний: {len(df)}")

# Показываем таблицу
print("\n📊 Таблица результатов:")
print(df[['filler', 'silane', 'amount', 'fp_before', 'ep_before', 'P_strength']].to_string())
