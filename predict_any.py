import sys
import pandas as pd
import joblib
from pathlib import Path

def load_models():
    models_dir = Path('models/')
    encoders_files = list(models_dir.glob('encoders_*.pkl'))
    if not encoders_files:
        raise FileNotFoundError('Модели не найдены!')
    
    latest = max(encoders_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace('encoders_', '')
    
    encoders = joblib.load(f'models/encoders_{timestamp}.pkl')
    scaler = joblib.load(f'models/scaler_{timestamp}.pkl')
    
    models = {}
    for f in models_dir.glob(f'*_{timestamp}.pkl'):
        name = f.stem.replace(f'_{timestamp}', '')
        if name not in ['encoders', 'scaler', 'feature_info']:
            models[name] = joblib.load(f)
    
    return models, encoders, scaler

def predict(filler_type, amount=30, silane=0):
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

if __name__ == '__main__':
    # Если переданы аргументы командной строки
    if len(sys.argv) > 1:
        filler = sys.argv[1]
        amount = float(sys.argv[2]) if len(sys.argv) > 2 else 30
        silane = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    else:
        # По умолчанию - шпинель 30 phr
        filler = 'шпинель'
        amount = 30
        silane = 0
    
    print(f'\n🔮 Предсказание для {filler} ({amount} phr, силан: {silane} phr)')
    print('=' * 50)
    
    result = predict(filler, amount, silane)
    for key, value in result.items():
        if abs(value) > 1e12:
            print(f'  {key}: {value:.3e}')
        else:
            print(f'  {key}: {value:.2f}')
